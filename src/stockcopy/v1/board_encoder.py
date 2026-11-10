import chess
import torch


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


def encode_board(board):
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

    return planes
