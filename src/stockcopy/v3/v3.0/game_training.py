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

DEBUG = True


device = torch.device('cuda')

# load model if available
model_path = Path('src/stockcopy/v3/model/stockcopy_v3.pt')
pgn_path = Path('src/stockcopy/v3/pgn')

model = algorithm.StockCopy().to(device)

# optimiser
optimiser = torch.optim.AdamW(
    model.parameters(),
    lr = 3e-6,
    weight_decay = 1e-4
)

# mcts
mcts = algorithm.MCTS(
    model = model,
    device = device,
    c_puct = 1.5,
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
        param['lr'] = 3e-6

    lichess_count = checkpoint.get('lichess_count', 0)
    game_count = checkpoint.get('game_count', 0)
    puzzle_count = checkpoint.get('puzzle_count', 0)




# start games
while True:
    print(f'game: {game_count}')

    board = chess.Board()

    saved_training_data = []

    model.eval()

    # game
    while (not board.is_game_over(claim_draw = True)) and len(saved_training_data) < 200:

        t0 = time.perf_counter()

        # save board
        saved_board = board.copy(stack = True)

        player = board.turn

        t1 = time.perf_counter()

        # run mcts
        root = mcts.search(
            board = board,
            num_simulations = 100
        )

        t2 = time.perf_counter()

        # obtain policy distribution
        search_policy = mcts.get_policy(
            root = root,
            temperature = 1
        )

        t3 = time.perf_counter()

        move = mcts.select_move(
            root = root,
            temperature = 1
        )

        t4 = time.perf_counter()

        selected_child = root.children[move]
    



        # save training infomation
        saved_training_data.append(
            {
                'board' : saved_board,
                'move' : move,
                'player' : player,
                'policy' : search_policy,

                'visits' : selected_child.visit_count,
                'q_value' : selected_child.q_value,
                'prior_value' : selected_child.prior_value,
                'nn_value' : root.value_prediction
            }
        )

        t5 = time.perf_counter()

        # play move
        board.push(move)

        if DEBUG:
            print(f'move: {len(saved_training_data)}')
            print(f'copy board: {(t1 - t0)*1000:.3f} ms')
            print(f'mcts search: {(t2 - t1)*1000:.3f} ms')
            print(f'mcts policy: {(t3 - t2)*1000:.3f} ms')
            print(f'mcts move: {(t4 - t3)*1000:.3f} ms')
            print(f'saving data: {(t5 - t4)*1000:.3f} ms')



    # game ended

    # check if over 200
    if not board.is_game_over(claim_draw = True):
        print('game cancelled due to bad data')
        continue

    game_count += 1

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

    position_count = len(saved_training_data)

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
    print(f'policy: {total_policy_loss / len(saved_training_data):.3f}')
    print(f'value: {total_value_loss / len(saved_training_data):.3f}')
    print(f'total loss: {total_loss / len(saved_training_data):.3f}')
    print(f'entropy: {total_entropy / len(saved_training_data):.3f}\n')




    #torch.save(
    #    {
    #        'model_state_dict' : model.state_dict(),
    #        'optimiser_state_dict' : optimiser.state_dict(),
    #        'lichess_count' : lichess_count,
    #        'puzzle_count' : puzzle_count,
    #        'game_count' : game_count
    #    },
    #    model_path
    #)



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
         