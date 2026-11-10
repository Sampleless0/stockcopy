import torch
import chess
import board_encoder


def move_to_index(move, board):
    from_square = move.from_square
    to_square = move.to_square

    if board.turn == chess.BLACK:
        from_square = chess.square(
            7 - chess.square_file(from_square),
            7 - chess.square_rank(from_square)
        )
        to_square = chess.square(
            7 - chess.square_file(to_square),
            7 - chess.square_rank(to_square)
        )

    df = chess.square_file(to_square) - chess.square_file(from_square)
    dr = chess.square_rank(to_square) - chess.square_rank(from_square)

    # queen directions
    directions = [
        (0, 1), (1, 1), (1, 0), (1, -1), 
        (0, -1), (-1, -1), (-1, 0), (-1, 1)
    ]

    # knight moves
    knight_moves = [
        (1, 2), (2, 1), (2, -1), (1, -2),
        (-1, -2), (-2, -1), (-2, 1), (-1, 2)
    ]

    move_type = None

    if move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):

        direction = -1 if df < 0 else (1 if df > 0 else 0)
        direction_index = {-1 : 0, 0 : 1, 1 : 2}[direction]

        piece_index = {
            chess.KNIGHT : 0,
            chess.BISHOP : 1,
            chess.ROOK : 2
        }[move.promotion]

        move_type = 64 + direction_index * 3 + piece_index

    elif (df, dr) in knight_moves:
        move_type = 56 + knight_moves.index((df, dr))

    else:
        distance = max(abs(df), abs(dr))

        if distance == 0 or not (
            df == 0 or dr == 0 or abs(df) == abs(dr)
        ):
            raise ValueError(f'Unsupported move: {move}')

        step = (df // distance, dr // distance)
        direction_index = directions.index(step)
        move_type = direction_index * 7 + (distance - 1)


    return from_square * 73 + move_type


def choose_nn_move(board, model, device = 'cuda'):
    x = board_encoder.encode_board(board).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output, value = model(x)

        logits = output[0]

        legal_moves = list(board.legal_moves)
        indices = []

        legal_moves = list(board.legal_moves)

    indices = [
        move_to_index(move, board)
        for move in legal_moves
    ]

    legal_indices = torch.tensor(
        indices,
        dtype=torch.long,
        device=logits.device
    )

    legal_logits = logits[legal_indices]
    probabilities = torch.softmax(legal_logits, dim=0)

    selected = torch.multinomial(
        probabilities,
        num_samples=1
    ).item()

    return legal_moves[selected], value.item()