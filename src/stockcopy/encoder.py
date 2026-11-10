import chess
import torch
from collections import deque
# setup board encoders

# convert chess.board to planes
# 20 planes

# current

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

# historical 1 (plane 12 - 23)
# historical 2 (plane 24 - 35)
# historical 3 (plane 36 - 47)
# historical 4 (plane 48 - 59)
# historical 5 (plane 60 - 71)
# historical 6 (plane 72 - 83)
# historical 7 (plane 84 - 95)

# plane 96: Friendly King Side Castle
# plane 97: Friendly Queen Side Castle
# plane 98: Enemy King Side Castle
# plane 99: Enemy Queen Side Castle

# plane 100: En passant target square
# plane 101: repeat twice
# plane 102: repeat thrice
# plane 103: how close to 50 move rule

def encode_piece_plane(board: chess.Board, perspective: chess.Color):
    planes = torch.zeros((12, 8, 8), dtype = torch.float32)
    
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
        if perspective == chess.BLACK:
            rank = 7 - rank

        is_friendly = piece.color == perspective

        offset = 0 if is_friendly else 6

        plane = offset + piece_offsets[piece.piece_type]

        planes[plane, rank, file] = 1

    return planes

def encode_auxillary_planes(board: chess.Board):
    planes = torch.zeros((8, 8, 8), dtype = torch.float32)

    # check castling (0-3)
    friendly = board.turn
    enemy = not board.turn

    if board.has_kingside_castling_rights(friendly):
        planes[0, :, :] = 1

    if board.has_queenside_castling_rights(friendly):
        planes[1, :, :] = 1

    if board.has_kingside_castling_rights(enemy):
        planes[2, :, :] = 1
    
    if board.has_queenside_castling_rights(enemy):
        planes[3, :, :] = 1

    # check en passant (4)
    if board.ep_square is not None and board.has_legal_en_passant():
        file = chess.square_file(board.ep_square)
        rank = chess.square_rank(board.ep_square)

        if board.turn == chess.BLACK:
            rank = 7 - rank

        planes[4, rank, file] = 1

    # check repetition (5-6)
    if board.is_repetition(2):
        planes[5, :, :] = 1

    if board.is_repetition(3):
        planes[6, :, :] = 1

    # check 50 move rule (7)
    moves = min(board.halfmove_clock, 100) / 100
    planes[7, :, :] = moves

    return planes


def encode_board(board: chess.Board) -> torch.Tensor:

    # main board (0-11)
    main_planes = encode_piece_plane(board, board.turn)

    # historical boards(12-95)
    historical_planes = []

    history_board = board.copy(stack = True)

    for _ in range(7):

        # stop if no more
        if not history_board.move_stack:
            break

        # go back one move
        history_board.pop()

        historical_planes.append(encode_piece_plane(history_board, board.turn))

    # ensure historical planes has 8 values
    while len(historical_planes) < 7:
        historical_planes.append(torch.zeros((12, 8, 8), dtype = torch.float32))

    # auxillary board (96-103)
    auxillary_planes = encode_auxillary_planes(board)

    planes = torch.cat(
        [
            main_planes,
            *historical_planes,
            auxillary_planes,
        ],
        dim = 0
    )

    return planes.unsqueeze(0)




# straight and diagonal moves (0-55)
# knight moves (56-63)
# underpromotion (64-72)


def encode_move(move: chess.Move, perspective: chess.Color) -> tuple[int, int, int]:

    # get start and end pos

    start_row = chess.square_rank(move.from_square)
    start_col = chess.square_file(move.from_square)

    end_row = chess.square_rank(move.to_square)
    end_col = chess.square_file(move.to_square)

    # check for black
    if perspective == chess.BLACK:
        start_row = 7 - start_row
        end_row = 7 - end_row

    # find distance
    dr = end_row - start_row
    dc = end_col - start_col

    # check for under promotion (64 - 72)
    if move.promotion in (
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK
    ):
        # underpromotions (64 - 72)

        promotions = (
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK
        )

        promotion_index = promotions.index(move.promotion)

        plane = 67 + (dc * 3) + promotion_index

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

        plane = (direction_index * 7) + distance - 1

    # knight moves (56 - 63)
    else:

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


def decode_move():
    pass


