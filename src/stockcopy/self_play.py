from pathlib import Path
from itertools import chain
import torch
import chess
import chess.engine
import random

import nnmodel
import mcts
import buffer
import analyse
import training
import saving

print('\nstarting self training')

# initialise stuff
model_path = Path(r'C:\Users\sampl\Documents\code\personal\stockcopy\src\stockcopy\model\test_model.pt')
stockfish_path = Path(r'C:\Users\sampl\Documents\code\personal\stockcopy\src\stockcopy\model\stockfish-windows-x86-64-universal.exe')
replay_path = Path(r'C:\Users\sampl\Documents\code\personal\stockcopy\src\stockcopy\buffers\replay_buffer.pt')




# setup models

model = nnmodel.StockCopy().to('cuda')

optimiser = torch.optim.AdamW(
    model.parameters(),
    lr = 0.0003,
    weight_decay = 0.0001
)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimiser,
    lambda step: 1
)

searcher = mcts.MCTS(model)

replay_buffer = buffer.ReplayBuffer(
    capacity = 1000000
)

play_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
play_engine.configure({'UCI_LimitStrength' : True, 'UCI_Elo' : 1320})

engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
engine.configure({'UCI_ShowWDL' : True, 'Threads' : 4, 'Hash' : 256})


game_count = 0

wins = 0
draws = 0
losses = 0



# load model
if model_path.exists():

    checkpoint = torch.load(
        model_path,
        map_location = 'cuda'
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

    game_count = checkpoint.get('game_count', 0)

    print('\nloaded model')



# load replay buffer
if replay_path.exists():

    replay_checkpoint = torch.load(
        replay_path,
        map_location = 'cpu'
    )

    replay_buffer.load_state_dict(replay_checkpoint)

    print(f'\nloaded replay buffer with {len(replay_buffer)} positions\n')

game_count += 1

# play games
bot_play_white = True
play_engine_percent = 0.8

# start games
try:
    while True:
        print(f'started game: {game_count}\n')

        board = chess.Board()

        root = None
        claimed_draw = False

        model.eval()

        saved_training_data = []

        move_number = 1

        # game
        while not board.is_game_over(claim_draw = True):

            # grab temp
            if game_count % 100 == 0:
                temperature = 0
            elif move_number < 20:
                temperature = 1
            elif move_number < 60:
                temperature = 0.5
            else:
                temperature = 0.1

            print(f'game: {game_count} | {move_number} | {play_engine_percent:.2f}')

            if (board.turn == chess.WHITE) == bot_play_white:

                # save board
                saved_board = board.copy(stack = True)
                
                player = board.turn

                # run mcts
                root = searcher.search(
                    board = board,
                    num_simulations = 200,
                    root = root,
                    use_root_noise = False if temperature == 0 else True
                )

                
                move = searcher.select_move(
                    root = root,
                    temperature = temperature
                )

                next_root = root.children[move]

                # play move

                root = next_root

                saved_training_data.append({
                    'board' : saved_board
                })

            else:

                if random.random() > play_engine_percent:
                    move = random.choice(list(board.legal_moves))
                else:
                    move = play_engine.play(board, chess.engine.Limit(nodes = 1)).move

                # advance root
                root = root.children.get(move) if root is not None else None


            board.push(move)

            move_number += 1
            
        # game ended
        print(f'\ngame ended: {game_count}')

        # check who won
        result = board.result()

        if result == '1-0':
            if bot_play_white:
                print('\nbot won')
                wins += 1
                play_engine_percent = min(1, play_engine_percent + 0.01)
            else:
                print('\nbot lost')
                losses += 1
                play_engine_percent = max(0, play_engine_percent - 0.01)

        elif result == '0-1':
            if bot_play_white:
                print('\nbot lost')
                losses += 1
                play_engine_percent = max(0, play_engine_percent - 0.01)
            else:
                print('\nbot won')
                wins += 1
                play_engine_percent = min(1, play_engine_percent + 0.01)
        else:
            print('\nbot drew')
            draws += 1
        
        print(f'\nwdl: {wins} | {draws} | {losses}')

        new_samples = analyse.prepare_training_samples(
            saved_training_data,
            engine
        )

        # add new positions to replay buffer
        replay_buffer.add_samples(new_samples)



        # analyse some samples
        if game_count % 20 == 0:

            # get number of steps
            eposhs = min(100, len(replay_buffer) // 128)

            print(eposhs, len(replay_buffer))

            for i in range(eposhs):

                # get replay buffer
                replay_size = min(512, len(replay_buffer))

                # get training samples
                training_samples = replay_buffer.sample(num_samples = replay_size)

                # train
                results = training.train_batch(model, optimiser, scheduler, training_samples, replay_size)

                print(f'policy ({results['policy_loss']:.3f}) | value ({results['value_loss']:.3f}) | entropy ({results['entropy']:.3f}) | total ({results['total_loss']:.3f})')


        # save model
        if game_count % 20 == 0:
            saving.save_model(model.state_dict(), optimiser.state_dict(), game_count, model_path)

            
        # save replay buffer
        if game_count % 200 == 0:
            saving.save_replay(replay_buffer, replay_path)


        # save pgn
        if game_count % 100 == 0:

            pgn_path = Path(r'C:\Users\sampl\Documents\code\personal\stockcopy\src\stockcopy\pgn')

            saving.save_pgn(board, game_count, bot_play_white, pgn_path)



        # save model
        if game_count % 1000 == 0:
            saving.save_checkpoint(model.state_dict(), optimiser.state_dict(), game_count)



        bot_play_white = not bot_play_white
        game_count += 1

finally:
    engine.quit()
    play_engine.quit()