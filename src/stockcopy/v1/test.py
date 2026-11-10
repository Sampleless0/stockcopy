import chess
import torch
import board_encoder
import nn_run
import nnmodel
import stockfih

device = torch.device('cuda')

model = nnmodel.ChessResNet(
    input_planes = 20,
    channels = 64
).to(device)

engine = chess.engine.SimpleEngine.popen_uci(
    'stockfish-windows-x86-64-universal.exe'
)

def play():
    board = chess.Board()

    move_number = 1

    print(board)

    while not board.is_game_over(claim_draw = True):

        move, value = nn_run.choose_nn_move(board, model)

        print('\nmove number: ', move_number, 'playing type: ', 'white' if board.turn else 'black')
        print(board.san(move))
        print(f'value: {value:+.3f}')

        result = stockfih.score_bot_move(board, move, engine)
        print('evaluation of move')
        print(result)

        board.push(move)

    print('game ended: ', board.result(claim_draw=True))

play()