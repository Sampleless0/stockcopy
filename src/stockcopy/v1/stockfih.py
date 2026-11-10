import math
import chess
import chess.engine

def evaluate_after_move(board, move, engine, depth=12):

    mover = board.turn

    board.push(move)
    try:
        info = engine.analyse(
            board,
            chess.engine.Limit(depth = depth)
        )

        score = info['score'].pov(mover)

        if score.is_mate():
            mate = score.mate()

            if mate is None:
                return 0.0

            return 100.0 - mate if mate > 0 else -100.0 - mate

        return score.score() / 100
    finally:
        board.pop()

def calculate_reward(best_score, bot_score, k=0.5):
    gap = max(0.0, best_score - bot_score)

    return math.exp(-k * gap) - 0.5

def score_bot_move(board, bot_move, engine, depth = 12):
    info = engine.analyse(
        board,
        chess.engine.Limit(depth = depth)
    )

    best_move = info['pv'][0]

    best_score = evaluate_after_move(
        board, best_move, engine, depth
    )
    bot_score = evaluate_after_move(
        board, bot_move, engine, depth
    )

    reward = calculate_reward(best_score, bot_score)

    return {
        'best_move' : best_move,
        'bot_move' : bot_move,
        'best_score' : best_score,
        'bot_score' : bot_score,
        'reward' : reward
    }