import chess
import torch
import torch.nn as nn
import torch.optim as optim
import chess.engine
import math
import random
import board_encoder
import nn_run
import nnmodel
import stockfih


device = torch.device('cuda')

model = nnmodel.ChessResNet(
    input_planes = 20,
    channels = 64
).to(device)

model.load_state_dict(
    torch.load('chess_model.pt', map_location = device, weights_only = True)
)

engine = chess.engine.SimpleEngine.popen_uci(
    'stockfish-windows-x86-64-universal.exe'
)

optimiser = optim.Adam(
    model.parameters(),
    lr = 1e-5
)

output_loss_func = nn.CrossEntropyLoss()
value_loss_func = nn.MSELoss()

game = 0

def play():

    global game

    board = chess.Board()

    move_number = 1

    total_loss_sum = 0
    total_reward = 0



    model.train()


    while not board.is_game_over(claim_draw = True):

        move, value = nn_run.choose_nn_move(board, model)

  
        result = stockfih.score_bot_move(board, move, engine)

        


        loss = train_step(
            board,
            bot_move = move,
            reward = result['reward']
        )

        
        total_loss_sum += loss
        total_reward += result['reward']


        value = random.randint(1, 100)

        if value <= 3:
            board.push(random.choice(list(board.legal_moves)))
        else:
            board.push(move)

        move_number += 1

    print('\n\n\ngame ended: ', board.result(claim_draw=True))

    positions = move_number - 1

    print('game: ', game, '     positions: ', positions)
    print('average loss', total_loss_sum / positions)
    print('average reward', total_reward / positions)

    game += 1

    torch.save(model.state_dict(), 'chess_model.pt')


def train_step(board, bot_move, reward):
    model.train()

    state = board_encoder.encode_board(board)
    x = state.unsqueeze(0).to(device).float()

    output_logits, predicted_value = model(x)

    legal_moves = list(board.legal_moves)
    
    # Get the indices of all legal moves
    legal_indices = [
        nn_run.move_to_index(move, board)
        for move in legal_moves
    ]

    # Keep only the logits for legal moves
    legal_logits = output_logits[0, legal_indices]

    log_prob = torch.nn.functional.log_softmax(
        legal_logits, dim = 0
    )

    bot_pos = legal_moves.index(bot_move)

    bot_log_prob = log_prob[bot_pos]

    loss = -0.1 * reward * bot_log_prob

    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimiser.step()

    return loss.item()




while True:
    play()
