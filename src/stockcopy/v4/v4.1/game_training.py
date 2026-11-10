import chess
import chess.engine
import chess.pgn
import chess.syzygy
import math
import torch
import torch.nn.functional as F
from pathlib import Path
import zstandard as zstd
import io
import time
import random

import algorithm 

# cd C:\Users\sampl\Documents\code\personal\stockcopy
# .venv\Scripts\activate
# python C:\Users\sampl\Documents\code\personal\stockcopy\src\stockcopy\v4\v4.1\game_training.py

print('\nprogram started!\n')

device = torch.device('cuda')

# load model if available
model_path = Path('src/stockcopy/v4/model/stockcopy_v4.pt')
pgn_path = Path('src/stockcopy/v4/pgn')
replay_buffer_path = Path('src/stockcopy/v4/model/replay_buffer.pt')
endgame_buffer_path = Path('src/stockcopy/v4/model/endgame_buffer.pt')
tablebase_path = Path('src/stockcopy/v4/syzygy')

model = algorithm.StockCopy().to(device)

# optimiser
optimiser = torch.optim.AdamW(
    model.parameters(),
    lr = 1e-4,
    weight_decay = 1e-4
)

# replay buffer
replay_buffer = algorithm.ReplayBuffer(
    capacity = 50000
)

# endgame buffer
endgame_buffer = algorithm.ReplayBuffer(
    capacity = 10000
)

# tablebase
tablebase = chess.syzygy.open_tablebase(
    str(tablebase_path),
    load_wdl = True,
    load_dtz = False
)

# mcts
mcts = algorithm.MCTS(
    model = model,
    device = device,
    c_puct = 1.5,
    batch_size = 8,
    virtual_loss = 1
)


lichess_count = 0
game_count = 0
puzzle_count = 0

wins = 0
draws = 0
losses = 0


# load model
if model_path.exists():

    checkpoint = torch.load(
        model_path,
        map_location = device
    )

    model.load_state_dict(
        checkpoint['model_state_dict']
    )

    optimiser.load_state_dict(
        checkpoint['optimiser_state_dict']
    )

    for param in optimiser.param_groups:
        param['lr'] = 1e-4

    lichess_count = checkpoint.get('lichess_count', 0)
    game_count = checkpoint.get('game_count', 0) + 1
    puzzle_count = checkpoint.get('puzzle_count', 0)

# load replay buffer
if replay_buffer_path.exists():

    replay_checkpoint = torch.load(
        replay_buffer_path,
        map_location = 'cpu'
    )

    replay_buffer.load_state_dict(
        replay_checkpoint
    )

    print(f'loaded replay buffer with {len(replay_buffer)} positions\n')

# load endgame buffer
if endgame_buffer_path.exists():

    endgame_checkpoint = torch.load(
        endgame_buffer_path,
        map_location = 'cpu'
    )

    endgame_buffer.load_state_dict(
        endgame_checkpoint
    )

    print(f'loaded endgame buffer with {len(endgame_buffer)} positions\n')

    


# start games
while True:
    print(f'started game: {game_count}\n')

    board = chess.Board()

    root = None

    claimed_draw = False

    saved_training_data = []
    new_endgame_samples = []
    seen_endgame_samples = set()

    model.eval()

    # game
    while not board.is_game_over(claim_draw = False):
        
        move_number = len(saved_training_data)
        temperature = (1 if move_number < 60 else 0.5)

        print(f'game: {game_count} | {move_number + 1}')

        # save board
        saved_board = board.copy(stack = True)
        
        player = board.turn

        # run mcts
        root = mcts.search(
            board = board,
            num_simulations = 800,
            root = root,
            use_root_noise = True
        )

        # check if wanna draw
        if board.can_claim_draw():

            visited_children = [
                child
                for child in root.children.values()
                if child.visit_count > 0
            ]

            if visited_children:
                # best continuation
                best_q = max(
                    child.q_value
                    for child in visited_children
                )

                draw_claim_threshold = 0.0001

                if best_q <= draw_claim_threshold:

                    claimed_draw = True
                    break

        # obtain policy distribution
        search_policy = mcts.get_policy(
            root = root,
            temperature = 1
        )

        key_board = saved_board.copy(stack = False)
        key_board.halfmove_clock = 0
        key_board.fullmove_number = 1

        position_key = key_board.fen()

        if position_key not in seen_endgame_samples:

            endgame_sample = algorithm.make_endgame_sample(
                board = saved_board,
                policy = search_policy,
                tablebase = tablebase
            )

            if endgame_sample is not None:
                new_endgame_samples.append(endgame_sample)



        move = mcts.select_move(
            root = root,
            temperature = temperature
        )

        next_root = root.children[move]


        # save training infomation
        saved_training_data.append(
            {
                'board' : saved_board,
                'move' : move,
                'player' : player,
                'policy' : search_policy,

                'visits' : next_root.visit_count,
                'q_value' : next_root.q_value,
                'prior_value' : next_root.prior_value,
                'nn_value' : root.value_prediction
            }
        )

        # play move
        board.push(move)

        root = next_root

        




    # game ended

    print(f'\nanalysing game: {game_count}')

    if claimed_draw:
        result = '1/2-1/2'
    else:
        result = board.result(claim_draw = False)

    # set result as white
    if result == '1-0':
        white_result = 1
        wins += 1
    elif result == '0-1':
        white_result = -1
        losses += 1
    else:
        white_result = 0
        draws += 1


    new_samples = algorithm.prepare_training_samples(
        saved_training_data,
        white_result
    )

    # add new positions to replay buffer
    replay_buffer.add_samples(new_samples)

    endgame_buffer.add_samples(new_endgame_samples)


    # prepare game training data
    for i in range(1, 6):

        # randomly sample old and new positions
        endgame_count = min(64, len(endgame_buffer))

        regular_count = min(448, len(replay_buffer))

        training_samples = replay_buffer.sample(num_samples = regular_count)

        if endgame_count > 0:
            training_samples += endgame_buffer.sample(num_samples = endgame_count)

        random.shuffle(training_samples)

        # train model using mini batches
        results = algorithm.train_self_play_game(
            model = model,
            optimiser = optimiser,
            samples = training_samples,
            device = device,
            batch_size = 32
        )

        print(f'\nresults for training {i}:')
        print(f'policy: {results['policy_loss']:.3f}')
        print(f'value: {results['value_loss']:.3f}')
        print(f'total loss: {results['total_loss']:.3f}')
        print(f'entropy: {results['entropy']:.3f}\n')

    print(f'winner: {result}')
    print(f'new positions: {len(new_samples)}\n')
    print(f'wins: {wins} | draws: {draws} | losses: {losses}')




    torch.save(
        {
            'model_state_dict' : model.state_dict(),
            'optimiser_state_dict' : optimiser.state_dict(),
            'lichess_count' : lichess_count,
            'puzzle_count' : puzzle_count,
            'game_count' : game_count
        },
        model_path
    )

    # save backup of model
    if game_count % 1000 == 0:
        torch.save(
            {
                'model_state_dict' : model.state_dict(),
                'optimiser_state_dict' : optimiser.state_dict(),
                'lichess_count' : lichess_count,
                'puzzle_count' : puzzle_count,
                'game_count' : game_count
            },
            Path(f'src/stockcopy/v4/model/stockcopy_v4_{game_count}.pt')
        )


    # save pgn
    if game_count % 10 == 0:
        # export pgn
        pgn_game = chess.pgn.Game()

        # headers
        pgn_game.headers['Event'] = 'stockcopy v4 vs stockcopy v4'
        pgn_game.headers['Site'] = 'local'
        pgn_game.headers['Date'] = '????.??.??'
        pgn_game.headers['Round'] = str(game_count)

        pgn_game.headers['White'] = 'stockcopy v4'
        pgn_game.headers['Black'] = 'stockcopy v4'

        pgn_game.headers['Result'] = result

        # add moves
        node = pgn_game

        for data in saved_training_data:

            move = data['move']

            node = node.add_variation(move)

            node.comment = (
                f"nn value: {data['nn_value']:+.3f} | "
                f"mcts q: {data['q_value']:+.3f} | "
                f"visits: {data['visits']} | "
                f"prior value: {data['prior_value']:.3f} | "
                f"policy: {data['policy'][move]:.1%} | "
            )

        # export to pgn txt
        exporter = chess.pgn.StringExporter(
            headers = True,
            variations = False,
            comments = True
        )

        pgn_text = pgn_game.accept(exporter)

        # save to file
        pgn_file_path = pgn_path / f'game_{game_count:08d}.pgn'

        with open(pgn_file_path, 'w', encoding = 'utf-8') as file:
            file.write(pgn_text)

        print('exported pgn')

    # save replay buffer
    if game_count % 10 == 0:
        # save replay buffer
        temp_path = replay_buffer_path.with_suffix('.tmp')

        torch.save(
            replay_buffer.state_dict(),
            temp_path
        )

        temp_path.replace(replay_buffer_path)

        print(f'saved replay buffer with {len(replay_buffer)} positions\n')
    
    # save endgame
    if game_count % 10 == 0:
        # save replay buffer
        temp_path = endgame_buffer_path.with_suffix('.tmp')

        torch.save(
            endgame_buffer.state_dict(),
            temp_path
        )

        temp_path.replace(endgame_buffer_path)

        print(f'saved endgame buffer with {len(endgame_buffer)} positions\n')
    

    game_count += 1
