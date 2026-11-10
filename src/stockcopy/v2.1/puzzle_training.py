import chess
import chess.engine
import chess.pgn
import math
import torch
import torch.nn.functional as F
from pathlib import Path
import zstandard as zstd
import io
import csv

import algorithm 


device = torch.device('cuda')

puzzle_path = Path('data/lichess_db_puzzle.csv.zst')

if not puzzle_path.exists():
    raise LookupError('lichess file not found')



# load model if available
model_path = Path('src/stockcopy/v2/model/stockcopy_v2.1.pt')

model = algorithm.StockCopy().to(device)

# load engine
engine = algorithm.load_engine()

# optimiser
optimiser = torch.optim.AdamW(
    model.parameters(),
    lr = 1e-5,
    weight_decay = 1e-4
)


lichess_count = 0

game_count = 0

current_puzzle_count = 0

puzzle_count = 0

debug = False


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
        param['lr'] = 1e-5

    lichess_count = checkpoint.get('lichess_count', 0)
    game_count = checkpoint.get('game_count', 0)
    puzzle_count = checkpoint.get('puzzle_count', 0)


model.train()

# load lichess file
with open(puzzle_path, 'rb') as compressed_file:
    dctx = zstd.ZstdDecompressor()

    # open compressed file
    with dctx.stream_reader(compressed_file) as reader:

        text_stream = io.TextIOWrapper(
            buffer = reader,
            encoding = 'utf-8-sig'
        )

        # loop through games
        print('loading puzzles...')

        csv_reader = csv.DictReader(text_stream)


        for row in csv_reader:

            # if already analysed
            if current_puzzle_count < puzzle_count:
                current_puzzle_count += 1
                continue

            puzzle_id = row['PuzzleId']

            print(f'puzzle: {current_puzzle_count}   id: {puzzle_id}')

            # load board
            board = chess.Board(row['FEN'])
            
            moves = [
                chess.Move.from_uci(move)
                for move in row['Moves'].split()
            ]

            board.push(moves[0])

            total_loss = []

            # all moves
            for index in range(1, len(moves), 2):

                legal_moves = list(board.legal_moves)
                
                solution_moves = [moves[index]]

                # check if checkmate move
                board_copy = board.copy()
                board_copy.push(moves[index])

                if board_copy.is_checkmate():

                    # add all solution moves
                    solution_moves = []

                    for move in legal_moves:
        
                        board_copy = board.copy()
                        board_copy.push(move)

                        if board_copy.is_checkmate():
                            solution_moves.append(move)


                # process results
                target = torch.zeros(
                    len(legal_moves),
                    dtype = torch.float32,
                    device = device
                )

                # calculate probability distribution
                probability = 1 / len(solution_moves)

                # update results
                for move in solution_moves:
                    solution_index = legal_moves.index(move)

                    target[solution_index] = probability


                # let nn find a move
                input_tensor = algorithm.encode_board(board).to(device)
                
                policy, value = model(input_tensor)


                legal_logits = []
                
                for move in legal_moves:
                    plane, row, col = algorithm.encode_move_tensor(board, move)
                    legal_logits.append(policy[0, plane, row, col])

                legal_logits = torch.stack(legal_logits)

                model_probs = F.log_softmax(legal_logits, dim = 0)

                policy_loss = -(target * model_probs).sum()

                # evaluate stockfish score
                info = engine.analyse(
                    board,
                    chess.engine.Limit(depth = 8)
                )

                score = info['score'].pov(board.turn)

                best_cp = score.score(mate_score = 1000)

                value_target = torch.tensor(
                    [[math.tanh(best_cp / 600)]],
                    dtype = torch.float32,
                    device = device
                )

                value_loss = F.smooth_l1_loss(value, value_target)

                loss = policy_loss + 0.5 * value_loss

                model_show = model_probs.exp()

                if debug:
                    print('policy loss: ', policy_loss)
                    print('value loss: ', 0.5 * value_loss)
                    print('loss: ', loss)


                total_loss.append(loss.detach().item())

                # training
                optimiser.zero_grad(set_to_none = True)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm = 1
                )

                optimiser.step()


                board.push(moves[index])

                # check if there is more
                if index + 1 < len(moves):

                    board.push(moves[index + 1])

            print(f'rewards for game {current_puzzle_count}')
            print(f'positions: {len(total_loss)}')
            print(f'avg loss: {sum(total_loss) / len(total_loss):.3f}   max loss: {max(total_loss):.3f}   min loss: {min(total_loss):.3f}')
            print('')

            current_puzzle_count += 1




            if current_puzzle_count % 100 == 0:
                print('Saving Model...')

            
                torch.save(
                    {
                        'model_state_dict' : model.state_dict(),
                        'optimiser_state_dict' : optimiser.state_dict(),
                        'lichess_count' : lichess_count,
                        'puzzle_count' : current_puzzle_count,
                        'game_count' : game_count
                    },
                    model_path
                )

                print('Saved Model!\n')


print('Saving LAST Model...')

                
torch.save(
    {
        'model_state_dict' : model.state_dict(),
        'optimiser_state_dict' : optimiser.state_dict(),
        'lichess_count' : lichess_count,
        'puzzle_count' : current_puzzle_count,
        'game_count' : game_count
    },
    model_path
)

print('Saved LAST Model!')