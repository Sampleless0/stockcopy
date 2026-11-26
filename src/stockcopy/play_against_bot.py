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

model_path = Path(r'C:\Users\sampl\Documents\code\personal\stockcopy\src\stockcopy\model\test_model.pt')


model = nnmodel.StockCopy().to('cuda')
searcher = mcts.MCTS(model)

# load model
if model_path.exists():

    checkpoint = torch.load(
        model_path,
        map_location = 'cuda'
    )

    model.load_state_dict(checkpoint['model_state_dict'])

    game_count = checkpoint.get('game_count', 0)

    print('\nloaded model')






board = chess.Board()

player_color = chess.WHITE if input('does player start? (y/n)') == 'y' else chess.BLACK

model.eval()

root = None
claimed_draw = False

while not board.is_game_over(claim_draw = True):

    # check if player move
    if board.turn == player_color:

        print(board)

        move = input('input legal move ->')

        board.push_uci(move)

        root = root.children.get(move) if root is not None else None

    else:
        # bot move
        player = board.turn
        
        # run mcts
        root = searcher.search(
            board = board,
            num_simulations = 4800,
            root = root,
            use_root_noise = False
        )

        move = searcher.select_move(
            root = root,
            temperature = 0
        )

        next_root = root.children[move]

        # play move

        root = next_root

        board.push(move)