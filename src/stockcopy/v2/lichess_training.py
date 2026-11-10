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
model_path = Path('src/stockcopy/v2/model/stockcopy_v2.pt')

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

    lichess_count = checkpoint.get('lichess_count', 0)
    game_count = checkpoint.get('game_count', 0)




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
            
            # analyse game
            board = chess.Board()

            # store info
            rewards = []

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

                # distribution
                distribution = torch.distributions.Categorical(
                    logits = legal_logits
                )

                # sample an action
                action = distribution.sample()

                model_move = legal_moves[action.item()]

                log_prob = distribution.log_prob(action)
                entropy = distribution.entropy()

                # now analyse move
                reward_cp = algorithm.calculate_reward(engine, board, model_move)

                reward = torch.tensor(
                    math.tanh(reward_cp),
                    dtype = value.dtype,
                    device = device
                )

                rewards.append(reward.detach().item())

                value_prediction = value.reshape(())

                advantage = reward - value_prediction.detach()

                policy_loss = -log_prob * advantage

                value_loss = F.smooth_l1_loss(value_prediction, reward)

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

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
            if rewards:
                print(sum(rewards) / len(rewards))
            print('')

            current_game_count += 1




            if current_game_count % 50 == 0:
                print('Saving Model...')

                
                torch.save(
                    {
                        'model_state_dict' : model.state_dict(),
                        'optimiser_state_dict' : optimiser.state_dict(),
                        'lichess_count' : current_game_count,
                        'game_count' : game_count
                    },
                    model_path
                )

                print('Saved Model!')


print('Saving LAST Model...')

                
torch.save(
    {
        'model_state_dict' : model.state_dict(),
        'optimiser_state_dict' : optimiser.state_dict(),
        'lichess_count' : current_game_count,
        'game_count' : game_count
    },
    model_path
)

print('Saved LAST Model!')




















with open(lichess_path, 'rb') as compressed_file:
    dctx = zstd.ZstdDecompressor()

    with dctx.stream_reader(compressed_file) as reader:
        text_stream = io.TextIOWrapper(
            reader,
            encoding = 'utf-8'
        )

        game = chess.pgn.read_game(text_stream)

        if game is not None:
            print('event:', game.headers.get('Event'))
            print(
                'Players:',
                game.headers.get('White'),
                'vs',
                game.headers.get('Black')
            )

            moves = list(game.mainline_moves())

            print('Number of moves:', len(moves))


board = game.board()
positions = []