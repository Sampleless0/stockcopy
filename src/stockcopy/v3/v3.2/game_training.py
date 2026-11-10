import chess
import chess.engine
import chess.pgn
import math
import torch
import torch.nn.functional as F
from pathlib import Path
import zstandard as zstd
import io
import time

import algorithm 

print('\nprogram started!\n')

device = torch.device('cuda')

# load model if available
model_path = Path('src/stockcopy/v3/model/stockcopy_v3.pt')
pgn_path = Path('src/stockcopy/v3/pgn')

model = algorithm.StockCopy().to(device)

# optimiser
optimiser = torch.optim.AdamW(
    model.parameters(),
    lr = 1e-3,
    weight_decay = 1e-4
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
        param['lr'] = 1e-3

    lichess_count = checkpoint.get('lichess_count', 0)
    game_count = checkpoint.get('game_count', 0) + 1
    puzzle_count = checkpoint.get('puzzle_count', 0)




# start games
while True:
    print(f'started game: {game_count}')

    board = chess.Board()

    root = None

    saved_training_data = []

    model.eval()

    # game
    while (not board.is_game_over(claim_draw = True)) and len(saved_training_data) < 200:
        
        move_number = len(saved_training_data)
        temperature = (1 if move_number < 30 else 0.1)


        # save board
        saved_board = board.copy(stack = True)
        
        player = board.turn

        # run mcts
        root = mcts.search(
            board = board,
            num_simulations = 400,
            root = root,
            use_root_noise = True
        )

        # obtain policy distribution
        search_policy = mcts.get_policy(
            root = root,
            temperature = temperature
        )

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

    # check if over 200
    if not board.is_game_over(claim_draw = True):
        print('game cancelled due to bad data\n')
        temperature = 1
        continue

    print(f'\nanalysing game: {game_count}')

    result = board.result(claim_draw = True)

    # set result as white
    if result == '1-0':
        white_result = 1
    elif result == '0-1':
        white_result = -1
    else:
        white_result = 0


    # analyse game
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_loss = 0

    optimiser.zero_grad(set_to_none = True)

    model.train()

    # review each position
    for data in saved_training_data:

        board = data['board']

        player = data['player']


        # change result based on pov
        if player == chess.WHITE:
            self_result = white_result
        else:
            self_result = -white_result

        self_result = torch.tensor(
            self_result,
            dtype = torch.float32,
            device = device
        )

        # forward pass
        legal_moves = list(board.legal_moves)

        # let nn find a move
        input_tensor = algorithm.encode_board(board).to(device)

        policy, value = model(input_tensor)

        legal_logits = []
        for move in legal_moves:
            plane, row, col = algorithm.encode_move_tensor(board, move)
            legal_logits.append(policy[0, plane, row, col])

        legal_logits = torch.stack(legal_logits)



        log_probs = F.log_softmax(
            legal_logits,
            dim = 0
        )

        probs = log_probs.exp()


        # retrieve mcts target distribution
        
        search_policy = data['policy']

        policy_target = torch.tensor(
            [
                search_policy[move] 
                for move in legal_moves
            ],
            dtype = torch.float32,
            device = device
        )

        # normalise target distribution

        policy_target = policy_target / policy_target.sum()

        # cross entropy for policy loss

        policy_loss = -(policy_target * log_probs).sum()

        # value loss
        value_prediction = value.squeeze()

        value_loss = F.smooth_l1_loss(
            value_prediction,
            self_result
        )

        # entropy
        entropy = -(probs * log_probs).sum()

        # total loss
        loss = policy_loss + 0.5 * value_loss

        position_count = len(saved_training_data)

        # backpropagate
        (loss / position_count).backward()


        total_policy_loss += policy_loss.detach().item()
        total_value_loss += value_loss.detach().item()
        total_entropy += entropy.detach().item()
        total_loss += loss.detach().item()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm = 1
    )

    optimiser.step()

    print('\nresults:')
    print(f'winner: {result}')
    print(f'moves: {len(saved_training_data)}')
    print(f'policy: {total_policy_loss / position_count:.3f}')
    print(f'value: {total_value_loss / position_count:.3f}')
    print(f'total loss: {total_loss / position_count:.3f}')
    print(f'entropy: {total_entropy / position_count:.3f}\n')




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
            Path(f'src/stockcopy/v3/model/stockcopy_v3_{game_count}.pt')
        )



    if game_count % 10 == 0:
        # export pgn
        pgn_game = chess.pgn.Game()

        # headers
        pgn_game.headers['Event'] = 'stockcopy v3 vs stockcopy v3'
        pgn_game.headers['Site'] = 'local'
        pgn_game.headers['Date'] = '????.??.??'
        pgn_game.headers['Round'] = str(game_count)

        pgn_game.headers['White'] = 'stockcopy v3'
        pgn_game.headers['Black'] = 'stockcopy v3'

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


    game_count += 1
