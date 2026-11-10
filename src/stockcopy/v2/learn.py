import chess
import chess.engine
import chess.pgn
import zstandard as zstd
import io
import numpy as np
import torch
import torch.nn as nn

board = chess.Board()

#print(board)
#print('Turn: ', board.turn)
#print('Legal moves: ', list(board.legal_moves))
#print('Number of legal moves: ', board.legal_moves.count())


def inspect_position(board):
    info = {
        'turn' : 'White' if  board.turn else 'Black',
        'legal_moves' : [move.uci() for move in board.legal_moves],
        'in_check' : board.is_check(),
        'game_over' : board.is_game_over()
    }

    return info

print(inspect_position(board))



def piece_at(board, square):
    index = chess.parse_square(square)
    piece_symbol = board.piece_at(index)

    if piece_symbol is None:
        return 'Empty'

    piece_to_name = {
        'P' : 'White Pawn',
        'N' : 'White Knight',
        'B' : 'White Bishop',
        'R' : 'White Rook',
        'Q' : 'White Queen',
        'K' : 'White King',
        'p' : 'Black Pawn',
        'n' : 'Black Knight',
        'b' : 'Black Bishop',
        'r' : 'Black Rook',
        'q' : 'Black Queen',
        'k' : 'Black King',
    }

    return piece_to_name[str(piece_symbol)]

    

print(piece_at(board, "e1"))
print(piece_at(board, "a7"))
print(piece_at(board, "e4"))


def white_pawn_plane(board):
    # initalise plane
    plane = np.zeros((8, 8), dtype = np.float32)

    for file in range(8):
        for rank in range(8):
            index = chess.square(file, rank)
            piece = board.piece_at(index)

            if piece == chess.Piece(chess.PAWN, chess.WHITE):
                plane[7 - rank][file] = 1

    return plane


print(white_pawn_plane(board))

board.push_uci("e2e4")
print(white_pawn_plane(board))


def black_pawn_plane(board):
    # initalise plane
    plane = np.zeros((8, 8), dtype = np.float32)

    for file in range(8):
        for rank in range(8):
            index = chess.square(file, rank)
            piece = board.piece_at(index)

            if piece == chess.Piece(chess.PAWN, chess.BLACK):
                plane[7 - rank][file] = 1

    return plane


print(white_pawn_plane(board))


print("White pawns:")
print(white_pawn_plane(board))

print("Black pawns:")
print(black_pawn_plane(board))

print("White pawn count:", white_pawn_plane(board).sum())
print("Black pawn count:", black_pawn_plane(board).sum())


def encode_piece_plane(board, piece_type, color):
    # initalise plane
    plane = np.zeros((8, 8), dtype = np.float32)

    for file in range(8):
        for rank in range(8):
            index = chess.square(file, rank)
            piece = board.piece_at(index)

            if piece == chess.Piece(piece_type, color):
                plane[7 - rank][file] = 1

    return plane


white_pawns = encode_piece_plane(board, chess.PAWN, chess.WHITE)
black_knights = encode_piece_plane(board, chess.KNIGHT, chess.BLACK)
white_kings = encode_piece_plane(board, chess.KING, chess.WHITE)

print("White pawns:", white_pawns.sum())
print("Black knights:", black_knights.sum())
print("White kings:", white_kings.sum())


def encode_board(board):
    planes = []
    pieces = [
        (chess.PAWN, chess.WHITE),
        (chess.KNIGHT, chess.WHITE),
        (chess.BISHOP, chess.WHITE),
        (chess.ROOK, chess.WHITE),
        (chess.QUEEN, chess.WHITE),
        (chess.KING, chess.WHITE),
        (chess.PAWN, chess.BLACK),
        (chess.KNIGHT, chess.BLACK),
        (chess.BISHOP, chess.BLACK),
        (chess.ROOK, chess.BLACK),
        (chess.QUEEN, chess.BLACK),
        (chess.KING, chess.BLACK)
    ]

    for piece_type, colour in pieces:
        planes.append(encode_piece_plane(board, piece_type, colour))

    numpy_stack = np.stack(planes)
    tensor = torch.from_numpy(numpy_stack)

    tensor = tensor.unsqueeze(0)

    return tensor



encoded = encode_board(board)

print(encoded.shape)


conv = nn.Conv2d(
    in_channels = 12,
    out_channels = 32,
    kernel_size = 3,
    padding = 1
)

relu = nn.ReLU()

batch = torch.randn(3, 12, 8, 8)

x = conv(batch)
x = relu(x)

print(x.shape)

block = nn.Sequential(
    nn.Conv2d(12, 32, kernel_size = 3, padding = 1),
    nn.ReLU()
)

x = block(batch)


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
        nn.ReLU()
    )


block1 = conv_block(12, 32)
block2 = conv_block(32, 64)


class ChessCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.policy_conv = nn.Conv2d(
            in_channels = 64,
            out_channels = 64, # supposed to be 73
            kernel_size = 1,
            padding = 0
        )   

        self.features = nn.Sequential(
            conv_block(12, 32),
            conv_block(32, 64),
            self.policy_conv
        )

    def forward(self, x):

        x = self.features(x)

        x = x.permute(0, 2, 3, 1)

        x = torch.flatten(x, start_dim = 1)


        return x


model = ChessCNN()
batch = torch.randn(3, 12, 8, 8)

output = model(batch)
print(output.shape)


scores = torch.tensor([0.2, 1.5, -0.3, 2.1, 0.7])
best_move_index = torch.argmax(scores)

print(best_move_index)

square_name = chess.square_name(2)

print(square_name)


index = 220
from_square = index // 73
move_type = index % 73

square_name = chess.square_name(from_square)
print(square_name)




board = chess.Board()

for move in board.legal_moves:
    print(move.uci())

print(board.legal_moves.count())




board = chess.Board()
legal_moves = list(board.legal_moves)
scores = torch.randn(len(legal_moves))
best_index = torch.argmax(scores)
best = legal_moves[best_index]

print(best)

print("Legal moves:", len(legal_moves))
print("Scores:", len(scores))









board = chess.Board()

x = encode_board(board)

model.eval()

with torch.no_grad():
    output = model(x)

print(output.shape)
print(output[0][:10])

best_index = torch.argmax(output)
print(best_index)

from_square = best_index // 73
move_type = best_index % 73

print(from_square)
print(move_type)


move = chess.Move.from_uci('e2e4')

print(move.from_square)
print(move.to_square)


index = move.from_square * 64 + move.to_square

print(index)

def encode_move(move):
    return move.from_square * 64 + move.to_square


move = chess.Move.from_uci("e2e4")
print(encode_move(move))  # 796

def decode_move(index):

    from_square = index // 64
    to_square = index % 64

    return chess.Move(from_square, to_square)



test_moves = ['e2e4', 'g1f3', 'b1c3', 'h2h4']

for uci in test_moves:
    move = chess.Move.from_uci(uci)

    # encode

    index = encode_move(move)

    # decode

    decoded = decode_move(index)

    # print

    print(uci, index, decoded.uci())


def square_to_tensor_index(square):
    file = chess.square_file(square)
    rank = chess.square_rank(square)

    tensor_index = (7 - rank) * 8 + file

    return tensor_index

e2 = chess.parse_square("e2")
e4 = chess.parse_square("e4")

print(square_to_tensor_index(e2))  # 52
print(square_to_tensor_index(e4))  # 36


def encode_move_tensor(move):
    start = move.from_square
    end = move.to_square

    start_tensor = square_to_tensor_index(start)
    end_tensor = square_to_tensor_index(end)

    tensor = start_tensor * 64 + end_tensor

    return tensor



move = chess.Move.from_uci("e2e4")
print(encode_move_tensor(move))



def decode_move_tensor(index):
    start_tensor = index // 64
    end_tensor = index % 64

    start_rank = 7 - (start_tensor // 8)
    start_file = start_tensor % 8

    end_rank = 7 - (end_tensor // 8)
    end_file = end_tensor % 8

    start = chess.square(start_file, start_rank)
    end = chess.square(end_file, end_rank)

    move = chess.Move(start, end)

    return move



test_moves = ["e2e4", "g1f3", "b1c3", "h2h4"]

for uci in test_moves:
    move = chess.Move.from_uci(uci)

    index = encode_move_tensor(move)
    decoded = decode_move_tensor(index)

    print(uci, index, decoded.uci())






model = ChessCNN()

board = chess.Board()

input_tensor = encode_board(board)

output = model(input_tensor)

print(output.shape)

legal_moves = list(board.legal_moves)

move_indices = [encode_move_tensor(move) for move in legal_moves]

legal_scores = output[0, move_indices]

best_index = torch.argmax(legal_scores).item()
best_move = legal_moves[best_index]

print(best_move)
print('is legal: ', board.is_legal(best_move))

board.push(best_move)
print(board)


input_tensor = encode_board(board)

output = model(input_tensor)

print(output.shape)

legal_moves = list(board.legal_moves)

move_indices = [encode_move_tensor(move) for move in legal_moves]

legal_scores = output[0, move_indices]

best_index = torch.argmax(legal_scores).item()
best_move = legal_moves[best_index]

print(best_move)
print('is legal: ', board.is_legal(best_move))

board.push(best_move)
print(board)


def choose_move(model, board):
    input_tensor = encode_board(board)
    output = model(input_tensor)

    legal_moves = list(board.legal_moves)

    move_indices = [
        encode_move_tensor(move)
        for move in legal_moves
    ]

    legal_scores = output[0, move_indices]

    best_index = torch.argmax(legal_scores).item()

    return legal_moves[best_index]




model.eval()
board = chess.Board()
max_moves = 200

with torch.inference_mode():
    for _ in range(max_moves):
        if board.is_game_over():
            break

        move = choose_move(model, board)
        board.push(move)

        print(move, end=" ")

outcome = board.outcome()

print("Outcome:", outcome)
print("Termination:", outcome.termination.name)

print("\nResult:", board.result() if board.is_game_over() else "Move limit reached")
    


board = chess.Board()

input_tensor = encode_board(board)

target_move = chess.Move.from_uci("e2e4")

target_index = encode_move_tensor(target_move)

target_index = torch.tensor([target_index], dtype = torch.long)



print(target_index)

print(target_index.shape)


criterion = nn.CrossEntropyLoss()

loss = criterion(model(input_tensor), target_index)

print(loss.item())




model.train()

optimiser = torch.optim.Adam(model.parameters(), lr = 0.001)

# clear old gradients

optimiser.zero_grad()

# calculate the prediction and loss

output = model(input_tensor)
loss = criterion(output, target_index)

# calculate gradients

loss.backward()

# update models weights

optimiser.step()










model.eval()

with torch.no_grad():
    updated_output = model(input_tensor)
    updated_loss = criterion(updated_output, target_index)

print("Updated loss:", updated_loss.item())










model.train()

for step in range(20):
    optimiser.zero_grad()

    output = model(input_tensor)
    loss = criterion(output, target_index)

    # Calculate gradients
    
    loss.backward()

    # Update weights
    
    optimiser.step()

    if (step + 1) % 5 == 0:
        print(f"Step {step + 1}: {loss.item():.4f}")




model.eval()

with torch.no_grad():
    output = model(input_tensor)
    final_loss = criterion(output, target_index)

print("Final loss:", final_loss.item())
















training_examples = []

board = chess.Board()

# Example 1: starting position -> e2e4
training_examples.append(
    (board.copy(), chess.Move.from_uci("e2e4"))
)

board.push_uci("e2e4")

# Example 2: Black to move
# Add the example for e7e5 here
training_examples.append(
    (board.copy(), chess.Move.from_uci("e7e5"))
)

board.push_uci("e7e5")

# Example 3: White to move
# Add the example for g1f3 here
training_examples.append(
    (board.copy(), chess.Move.from_uci("g1f3"))
)










board_batch = torch.cat([
    encode_board(position)
    for position, move in training_examples
], dim = 0)


target_indices = torch.tensor(
    [
        encode_move_tensor(move)
        for position, move in training_examples
    ],
    dtype = torch.long
)

print('Board batch: ', board_batch.shape)
print('Target indices: ', target_indices)














model.train()

optimiser.zero_grad()

# Predict scores for all three positions
batch_output = model(board_batch)
print("Output shape:", batch_output.shape)

# Compare predictions with all three target moves
loss = criterion(batch_output, target_indices)
print("Batch loss:", loss.item())

# Calculate gradients and update weights
loss.backward()
optimiser.step()











model.train()

for step in range(100):
    # 1. Clear old gradients
    optimiser.zero_grad()

    # 2. Predict scores for all three positions
    batch_output = model(board_batch)

    # 3. Calculate the batch loss
    loss = criterion(batch_output, target_indices)

    # 4. Calculate gradients
    loss.backward()

    # 5. Update the weights
    optimiser.step()

    if (step + 1) % 20 == 0:
        print(f"Step {step + 1}: {loss.item():.4f}")









model.eval()

with torch.no_grad():
    batch_output = model(board_batch)
    final_loss = criterion(batch_output, target_indices)

    predictions = torch.argmax(batch_output, dim=1)

    correct = (predictions == target_indices).sum().item()
    accuracy = correct / len(target_indices)

print("Final loss:", final_loss.item())
print("Target indices:   ", target_indices.tolist())
print("Predicted indices:", predictions.tolist())
print(f"Training accuracy: {accuracy:.0%}")









model.eval()

with torch.inference_mode():
    for position, target_move in training_examples:
        predicted_move = choose_move(model, position)

        print("Expected:", target_move.uci())
        print("Predicted:", predicted_move.uci())
        print("Legal:", position.is_legal(predicted_move))
        print()













        extension_board = chess.Board()

for uci in ["e2e4", "e7e5", "g1f3"]:
    extension_board.push_uci(uci)

# Example 4: Black to move
training_examples.append(
    (extension_board.copy(), chess.Move.from_uci("b8c6"))
)

extension_board.push_uci("b8c6")

# Example 5: White to move
training_examples.append(
    (extension_board.copy(), chess.Move.from_uci("f1c4"))
)

print("Number of examples:", len(training_examples))





board_batch = torch.cat([
    encode_board(position)
    for position, move in training_examples
], dim=0)

target_indices = torch.tensor(
    [
        encode_move_tensor(move)
        for position, move in training_examples
    ],
    dtype=torch.long
)

print("Board batch:", board_batch.shape)
print("Targets:", target_indices.shape)





model.train()

for step in range(100):
    optimiser.zero_grad()

    batch_output = model(board_batch)
    loss = criterion(batch_output, target_indices)

    loss.backward()
    optimiser.step()

    if (step + 1) % 20 == 0:
        print(f"Step {step + 1}: {loss.item():.4f}")











    model.eval()

with torch.no_grad():
    batch_output = model(board_batch)
    final_loss = criterion(batch_output, target_indices)
    predictions = batch_output.argmax(dim=1)

    correct = (predictions == target_indices).sum().item()
    accuracy = correct / len(target_indices)

print("Final loss:", final_loss.item())
print("Targets:    ", target_indices.tolist())
print("Predictions:", predictions.tolist())
print(f"Training accuracy: {accuracy:.0%}")





test_board = chess.Board()

test_examples = []

test_examples.append(
    (test_board.copy(), chess.Move.from_uci("d2d4"))
)

test_board.push_uci('d2d4')

test_examples.append(
    (test_board.copy(), chess.Move.from_uci("d7d5"))
)

test_board.push_uci('d7d5')

test_examples.append(
    (test_board.copy(), chess.Move.from_uci("c2c4"))
)


correct = 0

for position, target_move in training_examples:
    print("Target:", target_move.uci())

    # Your turn: print the number of legal moves
    legal_moves = position.legal_moves
    legal_moves_count = legal_moves.count()
    #print(legal_moves_count)

    #print('Legal' if target_move in legal_moves else 'Not Legal')

    # call model prediction
    move = choose_move(model, position)

    print('Predicted: ', move.uci())

    if move == target_move:
        correct += 1

print('correct: ', correct)
print('accuracy: ', correct / len(training_examples) * 100, '%')



position = test_examples[2][0]
target = test_examples[2][1]

input_tensor = encode_board(position)
output = model(input_tensor)

legal_moves = list(position.legal_moves)

move_indices = [
    encode_move_tensor(move)
    for move in legal_moves
]

legal_scores = output[0, move_indices]

top_5 = torch.topk(legal_scores, 5)
chosen = torch.argmax(legal_scores).item()




for index, score in zip(top_5.indices, top_5.values):
    move = legal_moves[index.item()]
    print(move.uci(), score.item())

target_index = legal_moves.index(target)
target_score = legal_scores[target_index].item()

gap = legal_scores[chosen].item() - target_score

print("Target move:", target.uci())
print("Target score:", target_score)
print("Best move:", legal_moves[chosen].uci())
print("Score gap:", gap)

rank = 0

for score in legal_scores:
    if score > target_score:
        rank += 1


print("Target rank:", rank + 1)







print('TRAINING EXAMPLE')

for position, target in test_examples:

    input_tensor = encode_board(position)
    output = model(input_tensor)

    legal_moves = list(position.legal_moves)

    move_indices = [
        encode_move_tensor(move)
        for move in legal_moves
    ]

    legal_scores = output[0, move_indices]

    top_5 = torch.topk(legal_scores, 5)
    chosen = torch.argmax(legal_scores).item()

    target_index = legal_moves.index(target)
    target_score = legal_scores[target_index].item()

    gap = legal_scores[chosen].item() - target_score

    print("Target move:", target.uci())
    print("Target score:", target_score)
    print("Best move:", legal_moves[chosen].uci())
    print("Score gap:", gap)

    rank = 0

    for score in legal_scores:
        if score > target_score:
            rank += 1


    print("Target rank:", rank + 1)




engine = chess.engine.SimpleEngine.popen_uci(
    'stockfish-windows-x86-64-universal.exe'
)



board = chess.Board()

result = engine.play(
    board,
    chess.engine.Limit(time=0.2)
)

print("Stockfish recommends:", result.move.uci())



board = chess.Board()
board.push_uci("d2d4")
board.push_uci("d7d5")
position = board.copy()


result = engine.play(
    position,
    chess.engine.Limit(time=0.2)
)

print("Stockfish recommends:", result.move.uci())




engine_examples = []

engine_examples.append(
    (position.copy(), result.move)
)

print("Position:", engine_examples[0][0])
print("Target move:", engine_examples[0][1].uci())






pgn_path = r'data\lichess_db_standard_rated_2026-09.pgn.zst'

with open(pgn_path, 'rb') as compressed_file:
    dctx = zstd.ZstdDecompressor()

    with dctx.stream_reader(compressed_file) as reader:
        text_stream = io.TextIOWrapper(
            reader,
            encoding = 'utf-8'
        )

        game = chess.pgn.read_game(text_stream)

        if game is not None:
            print('event:', game.headers.get('Event'))
            print(
                'Players:',
                game.headers.get('White'),
                'vs',
                game.headers.get('Black')
            )

            moves = list(game.mainline_moves())

            print('Number of moves:', len(moves))


board = game.board()
positions = []

for ply, move in enumerate(game.mainline_moves()):
    # Your turn:
    # If ply is divisible by 10, save a copy of the board.
    # Then push the move onto the board.

    if ply % 10 == 0:
        positions.append(board.copy())

    board.push(move)


print("Positions extracted:", len(positions))




engine_examples = []

for position in positions:

    result = engine.play(
        position,
        chess.engine.Limit(time=0.5)
    )

    engine_examples.append((position.copy(), result.move))

    print('stockfish recommends:', result.move.uci())

print(engine_examples[0][1].uci())







for example in engine_examples:
    board_batch = torch.cat([
        encode_board(position)
        for position, move in engine_examples
    ], dim=0)

    target_indices = torch.tensor(
    [
        encode_move_tensor(move)
        for position, move in engine_examples
    ], dtype=torch.long)

print("Board batch:", board_batch.shape)
print("Targets:", target_indices.shape)
