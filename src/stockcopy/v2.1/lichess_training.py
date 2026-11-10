import chess
import chess.engine
import chess.pgn
import math
import torch
import torch.nn.functional as F
from pathlib import Path
import zstandard as zstd
import io

import algorithm 


device = torch.device('cuda')

lichess_path = Path('data/lichess_db_standard_rated_2026-09.pgn.zst')

if not lichess_path.exists():
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

current_game_count = 0

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
with open(lichess_path, 'rb') as compressed_file:
    dctx = zstd.ZstdDecompressor()

    # open compressed file
    with dctx.stream_reader(compressed_file) as reader:

        text_stream = io.TextIOWrapper(
            buffer = reader,
            encoding = 'utf-8'
        )

        # loop through games
        while True:
            print(f'Game {current_game_count}: ')

            game = chess.pgn.read_game(text_stream)

            if game is None:
                break

            # if already analysed
            if current_game_count < lichess_count:
                current_game_count += 1
                continue

            # grab moves
            moves = list(game.mainline_moves())

            if not moves:
                print('game has no positions\n')
                current_game_count += 1
                continue

            
            # analyse game
            board = game.board()

            # store info
            total_loss = []

            for recorded_move in moves:

                # let nn find a move
                input_tensor = algorithm.encode_board(board).to(device)
                
                policy, value = model(input_tensor)

                legal_moves = list(board.legal_moves)

                legal_logits = []

                for move in legal_moves:
                    plane, row, col = algorithm.encode_move_tensor(board, move)
                    legal_logits.append(policy[0, plane, row, col])

                legal_logits = torch.stack(legal_logits)



                stockfish_cp = algorithm.evaluate_legal_moves(engine, board, legal_moves)

                stockfish_probs = algorithm.set_targets(legal_moves, stockfish_cp).to(device)



                model_probs = F.log_softmax(legal_logits, dim = 0)


                policy_loss = -(stockfish_probs * model_probs).sum()

                best_cp = max(stockfish_cp.values())

                value_target = torch.tensor(
                    [[math.tanh(best_cp / 600)]],
                    dtype = torch.float32,
                    device = device
                )

                value_loss = F.smooth_l1_loss(value, value_target)

                loss = policy_loss + 0.5 * value_loss

                model_show = model_probs.exp()

                if debug:
                    print('stockfish probs: ', stockfish_probs)
                    print('model probs: ', model_probs.exp())
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


                board.push(recorded_move)

            print(f'rewards for game {current_game_count}')
            print(f'positions: {len(moves)}')
            print(f'avg loss: {sum(total_loss) / len(total_loss):.3f}   max loss: {max(total_loss):.3f}   min loss: {min(total_loss):.3f}')
            print('')

            current_game_count += 1




            if current_game_count % 10 == 0:
                print('Saving Model...')

            
                torch.save(
                    {
                        'model_state_dict' : model.state_dict(),
                        'optimiser_state_dict' : optimiser.state_dict(),
                        'lichess_count' : current_game_count,
                        'puzzle_count' : puzzle_count,
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
        'lichess_count' : current_game_count,
        'puzzle_count' : puzzle_count,
        'game_count' : game_count
    },
    model_path
)

print('Saved LAST Model!')