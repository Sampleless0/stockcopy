import chess
import torch
import math
import encoder

def score_value(score):

    if score.is_mate():
        moves = score.mate()
        sign = 1 if moves > 0 else -1
        return sign * (10000 - (100 * abs(moves)))

    return score.score()

def prepare_training_samples(saved_training_data, stockfish):
    samples = []

    for data in saved_training_data:

        board = data['board']

        # get legal moves
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            continue

        # encode board
        encoded_board = encoder.encode_board(board).squeeze(0)

        # encode legal move indices
        legal_ids = [
            encoder.get_policy_index(move, board.turn)
            for move in legal_moves
        ]

        # get policy target
        infos = stockfish.analyse(board, chess.engine.Limit(nodes = 50000), multipv = 5)

        # policy
        scores = { 
            info['pv'][0] : score_value(info['score'].pov(board.turn))
            for info in infos
        }

        best = max(scores.values())

        # softmax policy
        weights = [
            math.exp((scores[move] - best) / 100) if move in scores else 0
            for move in legal_moves
        ]

        total = sum(weights)

        policy_target = torch.tensor(
            [w / total for w in weights],
            dtype = torch.float32
        )

        # value
        wdl = infos[0]['wdl'].pov(board.turn)

        value_target = torch.tensor(
            [wdl.wins / 1000, wdl.draws / 1000, wdl.losses / 1000],
            dtype = torch.float32
        )

        # add to samples
        samples.append({
            'board' : encoded_board,
            'legal_ids' : torch.tensor(
                legal_ids,
                dtype = torch.long
            ),
            'policy_target' : policy_target,
            'value_target' : value_target
        })

    return samples