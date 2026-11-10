import chess
import chess.engine

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path





# setup nn modules

class Residual_Block(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels = channels,
            out_channels = channels,
            kernel_size = 3,
            padding = 1,
            bias = False
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            in_channels = channels,
            out_channels = channels,
            kernel_size = 3,
            padding = 1,
            bias = False
        )

        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU()

    def forward(self, x):

        # set residual first
        residual = x

        # convolution layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # combine residual layer
        x = x + residual
        x = self.relu(x)        

        return x


class StockCopy(nn.Module):
    def __init__(self):
        super().__init__()


        self.conv1 = nn.Conv2d(
            in_channels = 20,
            out_channels = 64,
            kernel_size = 3,
            padding = 1,
            bias = False
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.residual_tower = nn.Sequential(
            *[Residual_Block(64) 
            for _ in range(6)]
        )

        self.policy = nn.Conv2d(
            in_channels = 64,
            out_channels = 73,
            kernel_size = 1,
            padding = 0,
        )

        self.value = nn.Sequential(
            nn.Conv2d(
                in_channels = 64,
                out_channels = 1,
                kernel_size = 1,
                padding = 0
            ),
            nn.Flatten(start_dim = 1),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
        self.relu = nn.ReLU()

    def forward(self, x):

        # initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # residual tower
        x = self.residual_tower(x)

        # output values
        policy = self.policy(x)
        value = self.value(x)

        return policy, value






# setup board encoders

# convert chess.board to planes
# 20 planes

# plane 0: Friendly King
# plane 1: Friendly Queen
# plane 2: Friendly Rook
# plane 3: Friendly Bishop
# plane 4: Friendly Knight
# plane 5: Friendly Pawn

# plane 6: Enemy King
# plane 7: Enemy Queen
# plane 8: Enemy Rook
# plane 9: Enemy Bishop
# plane 10: Enemy Knight
# plane 11: Enemy Pawn

# plane 12: Friendly King Side Castle
# plane 13: Friendly Queen Side Castle
# plane 14: Enemy King Side Castle
# plane 15: Enemy Queen Side Castle

# plane 16: En passant target square
# plane 17: repeat twice
# plane 18: repeat thrice
# plane 19: how close to 50 move rule

def encode_board(board: chess.Board) -> torch.tensor:
    planes = torch.zeros((20, 8, 8), dtype = torch.float32)

    # piece to plane offset
    piece_offsets = {
        chess.KING : 0,
        chess.QUEEN : 1,
        chess.ROOK : 2,
        chess.BISHOP : 3,
        chess.KNIGHT : 4,
        chess.PAWN : 5
    }

    # fill each piece square (0-11)
    for square, piece in board.piece_map().items():
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        # rotate board if black
        if board.turn == chess.BLACK:
            file = 7 - file
            rank = 7 - rank

        is_friendly = piece.color == board.turn

        offset = 0 if is_friendly else 6

        plane = offset + piece_offsets[piece.piece_type]

        planes[plane, rank, file] = 1.0

    # check castling (12-15)
    friendly = board.turn
    enemy = not board.turn

    if board.turn == chess.WHITE:
        rights = [
            board.has_kingside_castling_rights(friendly),
            board.has_queenside_castling_rights(friendly),
            board.has_kingside_castling_rights(enemy),
            board.has_queenside_castling_rights(enemy),
        ]
    else:
        rights = [
            board.has_queenside_castling_rights(friendly),
            board.has_kingside_castling_rights(friendly),
            board.has_queenside_castling_rights(enemy),
            board.has_kingside_castling_rights(enemy),
        ]

    for i, has_right in enumerate(rights):
        if has_right:
            planes[12 + i, :, :] = 1.0

    # check en passant (16)
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        rank = chess.square_rank(board.ep_square)

        if board.turn == chess.BLACK:
            file = 7 - file
            rank = 7 - rank

        planes[16, rank, file] = 1.0

    # check repetition (17-18)
    if board.is_repetition(2):
        planes[17, :, :] = 1.0

    if board.is_repetition(3):
        planes[18, :, :] = 1.0

    # check 50 move rule (19)
    moves = min(board.halfmove_clock, 100) / 100
    planes[19, :, :] = moves

    return planes.unsqueeze(0)


def decode_board():
    pass







# encode pieces

def find_pos(board, square):
    row = chess.square_rank(square)
    col = chess.square_file(square)

    # check for black
    if board.turn == chess.BLACK:
        row = 7 - row
        col = 7 - col

    return row, col

# straight and diagonal moves (0-55)
# knight moves (56-63)
# underpromotion (64-72)

def encode_move_tensor(board: chess.Board, move: chess.Move):

    # find a start and end
    start_row, start_col = find_pos(board, move.from_square)
    end_row, end_col = find_pos(board, move.to_square)

    dr = end_row - start_row
    dc = end_col - start_col


    if move.promotion in (
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK
    ):

        # underpromotion (64-72)
        promotions = {
            chess.KNIGHT : 0,
            chess.BISHOP : 1,
            chess.ROOK : 2
        }

        promotion_index = promotions[move.promotion]

        plane = 67 + dc * 3 + promotion_index

        return plane, start_row, start_col

    # straight and diagonal moves (0-55)
    elif dr == 0 or dc == 0 or abs(dr) == abs(dc):
        directions = (
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1)
        )


        direction = (
            0 if dr == 0 else (1 if dr > 0 else -1),
            0 if dc == 0 else (1 if dc > 0 else -1)
        )

        direction_index = directions.index(direction)

        distance = max(abs(dr), abs(dc))

        plane = direction_index * 7 + distance - 1

        return plane, start_row, start_col

    else:
        
        # knight moves (56-63)
        knight_moves = (
            (2, 1),
            (1, 2),
            (-1, 2),
            (-2, 1),
            (-2, -1),
            (-1, -2),
            (1, -2),
            (2, -1)
        )

        knight_index = knight_moves.index((dr, dc))

        plane = 56 + knight_index

        return plane, start_row, start_col



# stockfish

def load_engine():
    stockfish_path = Path('stockfish-windows-x86-64-universal.exe')

    # check for path
    if not stockfish_path.exists():
        raise LookupError('stockfih not found')

    engine = chess.engine.SimpleEngine.popen_uci(
        stockfish_path
    )

    return engine

def calculate_reward(engine, board, move):

    # get stockfish best move
    best_info = engine.analyse(
        board = board,
        limit = chess.engine.Limit(depth = 8)
    )

    best_move = best_info['pv'][0]
    best_score = best_info['score'].pov(board.turn)


    # get model move
    model_info = engine.analyse(
        board = board,
        limit = chess.engine.Limit(depth = 8),
        root_moves = [move]
    )

    model_score = model_info['score'].pov(board.turn)


    best_value = best_score.score(mate_score = 1000)
    model_value = model_score.score(mate_score = 1000)

    reward = (model_value - best_value) / 200

    return reward

def evaluate_legal_moves(engine, board, legal_moves):
    scores = {}


    for move in legal_moves:
        info = engine.analyse(
            board = board,
            limit = chess.engine.Limit(depth = 8),
            root_moves = [move]
        )

        score = info['score'].pov(board.turn)

        scores[move] = score.score(mate_score = 1000)

    return scores

def set_targets(legal_moves, scores):

    scores_cp = torch.tensor(
        [
            scores[move]
            for move in legal_moves
        ],
        dtype = torch.float32
    )

    return F.softmax(scores_cp / 100, dim = 0)



# model predict
def predict(model, board):

    # find input tensor

    input_tensor = encode_board(board)

    policy, value = model(input_tensor)

    legal_moves = list(board.legal_moves)

    legal_scores = []

    for move in legal_moves:
        plane, row, col = encode_move_tensor(board, move)

        score = policy[0, plane, row, col]

        legal_scores.append(score)

    predicted_index = torch.argmax(torch.stack(legal_scores)).item()

    predicted_move = legal_moves[predicted_index]









    return predicted_move, value




