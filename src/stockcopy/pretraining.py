import chess
import chess.pgn
import torch
from pathlib import Path

import encoder

def get_white_result(result):
    results = {
        '1-0' : 1,
        '1/2-1/2' : 0,
        '0-1' : -1
    }

    return results[result]

def get_policy_index(move, perspective):

    plane, row, col = encoder.encode_move(move, perspective)

    return plane * 64 + row * 8 + col

def game_to_sample(game):

    moves = game.mainline_moves()

    board = game.board()

    white_result = get_white_result(game.headers.get('Result'))

    for played_move in moves:

        # get all legal moves
        legal_moves = list(board.legal_moves)

        # encode
        encoded_board = encoder.encode_board(board).squeeze(0)

        legal_indices = [
            get_policy_index(move, board.turn)
            for move in legal_moves
        ]

        # one hot policy target
        policy_target = torch.zeros(
            len(legal_moves),
            dtype = torch.float32
        )

        # find played move
        played_index = legal_moves.index(played_move)

        # set played move to 1
        policy_target[played_index] = 1

        # calculate value from current player pov
        if board.turn == chess.WHITE:
            value_target = white_result
        else:
            value_target = -white_result

        yield {
            'board' : encoded_board.cpu(),
            'legal_indices' : torch.tensor(
                legal_indices,
                dtype = torch.long
            ),
            'policy_target' : policy_target,
            'value_target' : value_target
        }

        board.push(played_move)

def iterate_pgn_sample(pgn_path, split, validation_every = 10):

    game_index = 0

    with pgn_path.open('r', encoding = 'utf-8') as pgn_file:

        while True:

            game = chess.pgn.read_game(pgn_file)
           
            if game is None:
                break

            # split games into train and validation
            is_validation = game_index % validation_every == 0
            game_index += 1

            if split == 'train' and is_validation:
                continue

            if split == 'val' and not is_validation:
                continue

    
            yield from game_to_sample(game)