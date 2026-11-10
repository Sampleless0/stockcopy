from __future__ import annotations

import chess
import chess.engine
import chess.syzygy

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import math
import random

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

        self.policy = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 73, kernel_size = 1)
        )

        self.value = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(start_dim = 1),
            nn.Linear(32 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
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



# mcts still learning so quite bad
class Node:

    def __init__(self, move = None, prior_value: float = 1):

        self.move = move

        self.prior_value = prior_value

        self.root_prior_value = None

        self.visit_count = 0
        self.value_sum = 0

        self.children: dict[chess.Move, Node] = {}

        self.expanded = False

        self.value_prediction = None

    @property
    def q_value(self):

        if self.visit_count == 0:
            return 0

        return self.value_sum / self.visit_count


    # adding a child (branching)
    def add_child(self, move: chess.Move, prior_value: float):

        child = Node(move, prior_value)

        self.children[move] = child

        return child

    # select child
    def select_child(self, c_puct = 1.5, use_root_prior = False):

        best_score = float('-inf')

        best_move = None
        best_child = None

        parent_visits = self.visit_count

        sqrt_parent = math.sqrt(parent_visits + 1)

        for move, child in self.children.items():

            # use noisy prior only when selecting from current root
            if use_root_prior and child.root_prior_value is not None:
                prior = child.root_prior_value
            else:
                prior = child.prior_value

            # add first play urgency
            fpu_reduction = 0.25

            parent_value = self.value_prediction if self.value_prediction is not None else 0

            # exploitation
            q = child.q_value if child.visit_count > 0 else parent_value - fpu_reduction

            # exploration
            u = c_puct * prior * sqrt_parent / (1 + child.visit_count)

            # calculate PUCT score
            score = q + u

            # find highest
            if score > best_score:

                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child, best_score

def terminal_value(board: chess.Board):

    outcome = board.outcome(claim_draw = True)

    # draw
    if outcome.winner is None:
        return 0

    # win / loss
    return 1 if outcome.winner == board.turn else -1

def get_policy_index(board, move):

    plane, row, col = encode_move_tensor(board, move)

    return plane * 64 + row * 8 + col

def evaluate_and_expand_batch(nodes: list[Node], boards: list[chess.Board], model, device):

    values = {}

    active_nodes = []
    active_boards = []

    for node, board in zip(nodes, boards):

        if board.is_game_over(claim_draw = False):
            node.expanded = True

            value = terminal_value(board)

            node.value_prediction = value

            values[id(node)] = value

        else:

            active_nodes.append(node)
            active_boards.append(board)



    # nothing to evaluate
    if not active_nodes:
        return values

    input_tensors = [
        encode_board(board)
        for board in active_boards
    ]

    # combine into one batch
    input_tensor = torch.cat(
        input_tensors,
        dim = 0
    ).to(device)

    # generate legal moves
    legal_moves_batch = [
        list(board.legal_moves)
        for board in active_boards
    ]

    batch_size = len(active_nodes)

    max_legal_moves = max(
        len(moves)
        for moves in legal_moves_batch
    )

    # prepare padded legal move indices
    legal_indices = torch.zeros(
        (batch_size, max_legal_moves),
        dtype = torch.long,
        device = device
    )

    legal_mask = torch.zeros(
        (batch_size, max_legal_moves),
        dtype = torch.bool,
        device = device
    )

    for i, (board, moves) in enumerate(zip(active_boards, legal_moves_batch)):

        indices = [
            get_policy_index(board, move)
            for move in moves
        ]

        count = len(indices)

        legal_indices[i, :count] = torch.tensor(
            indices,
            dtype = torch.long,
            device = device
        )

        legal_mask[i, :count] = True


    # nn evaluation
    prior_batch, value_batch = evaluate_policy_batch(
        model = model,
        input_tensor = input_tensor,
        legal_indices = legal_indices,
        legal_mask = legal_mask,
        device = device
    )

    # expand each position
    for i, node in enumerate(active_nodes):

        legal_moves = legal_moves_batch[i]
        priors = prior_batch[i][:len(legal_moves)]
                    
        for move, prior in zip(legal_moves, priors):

            node.add_child(
                move = move,
                prior_value = prior
            )

        node.expanded = True

        node.value_prediction = value_batch[i]

        values[id(node)] = value_batch[i]


    return values

def evaluate_and_expand_encoded_batch(nodes: list[Node], encoded_positions, legal_moves_batch, legal_ids_batch, model, device):

    values = {}

    if not nodes:
        return values

    input_tensor = torch.stack(
        encoded_positions
    ).to(device)

    batch_size = len(nodes)

    max_legal_moves = max(
        len(moves)
        for moves in legal_moves_batch
    )

    legal_indices = torch.zeros(
        (batch_size, max_legal_moves),
        dtype = torch.long
    )

    legal_mask = torch.zeros(
        (batch_size, max_legal_moves),
        dtype = torch.bool
    )

    # prepare legal move indices
    for i, indices in enumerate(legal_ids_batch):

        count = len(indices)

        legal_indices[i, :count] = torch.tensor(
            indices,
            dtype = torch.long
        )

        legal_mask[i, :count] = True

    # transfer to gpu
    legal_indices = legal_indices.to(device)
    legal_mask = legal_mask.to(device)


    # nn evaluation
    prior_batch, value_batch = evaluate_policy_batch(
        model = model,
        input_tensor = input_tensor,
        legal_indices = legal_indices,
        legal_mask = legal_mask,
        device = device
    )

    # expand each position
    for i, node in enumerate(nodes):

        legal_moves = legal_moves_batch[i]
        priors = prior_batch[i][:len(legal_moves)]
                    
        for move, prior in zip(legal_moves, priors):

            node.add_child(
                move = move,
                prior_value = prior
            )

        node.expanded = True

        node.value_prediction = value_batch[i]

        values[id(node)] = value_batch[i]


    return values
  



def backpropagate(path: list[Node], leaf_value: float):

    # go from leaf to root
    for node in reversed(path[1:]):

        # change perspective
        leaf_value = -leaf_value * 0.995

        # update stats
        node.visit_count += 1
        node.value_sum += leaf_value

    # update root visit count
    path[0].visit_count += 1

def apply_virtual_loss(path: list[Node], virtual_loss = 1):

    # temp increase root visits
    path[0].visit_count += 1

    # temp penalise selected branches
    for node in path[1:]:

        node.visit_count += 1

        node.value_sum -= virtual_loss

def revert_virtual_loss(path: list[Node], virtual_loss = 1):

    # temp increase root visits
    path[0].visit_count -= 1

    # temp penalise selected branches
    for node in path[1:]:

        node.visit_count -= 1

        node.value_sum += virtual_loss

def evaluate_policy_batch(model, input_tensor, legal_indices, legal_mask, device):

    with torch.inference_mode():

        # select precision
        amp_dtype = torch.bfloat16

        with torch.autocast(device_type = 'cuda', dtype = amp_dtype):

            policy_batch, value_batch = model(input_tensor)

        # flatten policy output
        policy_flat = policy_batch.flatten(start_dim = 1)

        # get legal move logits
        legal_logits = policy_flat.gather(dim = 1, index = legal_indices)

        # convert to fp32 before softmax
        legal_logits = legal_logits.float()

        # mask padded positions
        legal_logits = legal_logits.masked_fill(
            ~legal_mask,
            float('-inf')
        )

        # calculate probs in fp32
        prior_batch = F.softmax(
            legal_logits,
            dim = 1
        )

        # return results to cpu
        prior_batch = prior_batch.cpu().tolist()

        wdl = F.softmax(value_batch.float(), dim = 1)
        value_batch = (wdl[:, 0] - wdl[:, 2]).cpu().tolist()


    return prior_batch, value_batch

def prepare_training_samples(saved_training_data, white_result):
    samples = []

    for data in saved_training_data:

        board = data['board']

        # get legal moves
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            continue

        # encode board
        encoded_board = encode_board(board).squeeze(0)

        # encode legal move indices
        legal_ids = [
            get_policy_index(board, move)
            for move in legal_moves
        ]

        # get mcts policy target
        policy_target = torch.tensor(
            [
                data['policy'].get(move, 0)
                for move in legal_moves
            ],
            dtype = torch.float32
        )

        target_sum = policy_target.sum()

        if target_sum <= 0:
            continue

        # normalise
        policy_target /= target_sum

        # calc value target from player perspective
        player = data['player']

        if player == chess.WHITE:
            value_target = white_result
        else:
            value_target = -white_result

        # positions that is determined (loss or win)
        policy_weight = 0.2 if abs(data['q_value']) > 0.9 else 1

        samples.append({
            'board' : encoded_board,

            'legal_ids' : torch.tensor(
                legal_ids,
                dtype = torch.long
            ),

            'policy_target' : policy_target,
            'policy_weight' : policy_weight,

            'value_target' : float(value_target)
        })

    return samples

def train_self_play_game(model, optimiser, samples, device, batch_size = 32):
    position_count = len(samples)

    model.train()

    optimiser.zero_grad(set_to_none = True)

    # shuffle training positions
    order = torch.randperm(position_count).tolist()

    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_loss = 0

    # process mini-batches
    for start in range(0, position_count, batch_size):

        batch_indices = order[start:start + batch_size]

        batch = [
            samples[i]
            for i in batch_indices
        ]

        current_batch_size = len(batch)

        # prepare board inputs
        inputs = torch.stack([
            sample['board']
            for sample in batch
        ]).to(device)

        # find max number of legal moves
        max_legal_moves = max(
            len(sample['legal_ids'])
            for sample in batch
        )

        # prepare padded tensors on cpu
        legal_indices = torch.zeros(
            (current_batch_size, max_legal_moves),
            dtype = torch.long
        )

        legal_mask = torch.zeros(
            (current_batch_size, max_legal_moves),
            dtype = torch.bool
        )

        policy_targets = torch.zeros(
            (current_batch_size, max_legal_moves),
            dtype = torch.float32
        )

        value_targets = torch.tensor(
            [
                sample['value_target']
                for sample in batch
            ],
            dtype = torch.float32,
            device = device
        )

        # fill variable length legal move data
        for i, sample in enumerate(batch):
            count = len(sample['legal_ids'])

            legal_indices[i, :count] = sample['legal_ids']

            legal_mask[i, :count] = True

            policy_targets[i, :count] = sample['policy_target']

        # transfer padded tensor to gpu
        legal_indices = legal_indices.to(device)
        legal_mask = legal_mask.to(device)
        policy_targets = policy_targets.to(device)

        # nn forward
        policy_output, value_output = model(inputs)

        # flatten policy output
        policy_flat = policy_output.flatten(start_dim = 1)

        # select legal move logits
        legal_logits = policy_flat.gather(
            dim = 1,
            index = legal_indices
        )

        # mask padded moves
        legal_logits = legal_logits.masked_fill(
            ~legal_mask,
            float('-inf')
        )

        # policy probs
        log_probs = F.log_softmax(
            legal_logits,
            dim = 1
        )

        # avoid 0 * -inf
        safe_log_probs = log_probs.masked_fill(
            ~legal_mask,
            0
        )

        probs = log_probs.exp()

        policy_weights = torch.tensor(
            [
                sample['policy_weight']
                for sample in batch
            ],
            dtype = torch.float32,
            device = device
        )

        # policy loss for each position
        policy_losses = -(policy_targets * safe_log_probs).sum(dim = 1)

        policy_loss = (policy_losses * policy_weights).sum() / policy_weights.sum().clamp(min = 1e-6)

        # value loss
        value_classes = (1 - value_targets).round().long()
        value_loss = F.cross_entropy(value_output.float(), value_classes)

        # entropy
        entropy = -(probs * safe_log_probs).sum(dim = 1).mean()

        # combined loss
        loss = policy_loss + 0.5 * value_loss

        # accumulate gradients
        # weight by batch size so the final gradient represents the average over the whole game
        weighted_loss = (loss * current_batch_size / position_count)

        weighted_loss.backward()

        # record stats
        total_policy_loss += (policy_loss.detach().item() * current_batch_size)

        total_value_loss += (value_loss.detach().item() * current_batch_size)

        total_entropy += (entropy.detach().item() * current_batch_size)

        total_loss += (loss.detach().item() * current_batch_size)

    # update model once all mini batches
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm = 1
    )

    optimiser.step()

    return {
        'policy_loss' : total_policy_loss / position_count,
        'value_loss' : total_value_loss / position_count,
        'entropy' : total_entropy / position_count,
        'total_loss' : total_loss / position_count
    }


# mcts itself
class MCTS:

    def __init__(self, model, device, c_puct = 1.5, batch_size = 8, virtual_loss = 1):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.virtual_loss = virtual_loss


    # search
    def search(self, board: chess.Board, num_simulations = 800, root: Node = None, use_root_noise = False):
        self.model.eval()

        # create root
        if root is None:
            root = Node()

        scratch_board = board.copy(stack = True)

        # inital expansion (only if necessary)
        if not root.expanded or not root.children:
            root_values = evaluate_and_expand_batch([root], [board], self.model, self.device)

            root.value_prediction = root_values[id(root)]

        # reset temp root priors
        for child in root.children.values():
            child.root_prior_value = None

        # applies noise once per search
        if use_root_noise:
            self.add_root_noise(root, alpha = 0.3, epsilon = 0.25)

        # run simulations
        for batch_start in range(0, num_simulations, self.batch_size):
            current_batch_size = min(self.batch_size, num_simulations - batch_start)

            paths = []

            # collect unique leaf nodes
            leaf_nodes = {}


            for _ in range(current_batch_size):

                node = root

                path = [root]

                while node.expanded and node.children:
                
                    move, child, score = node.select_child(c_puct = self.c_puct, use_root_prior = (node is root))
    
                    node = child
    
                    path.append(node)

                # reserve this path
                apply_virtual_loss(path, self.virtual_loss)

                paths.append(path)

                # store unique leaf
                leaf_nodes[id(node)] = (node, path.copy())




            # batch eval

            batch_nodes = []

            encoded_positions = []
            legal_moves_batch = []
            legal_ids_batch = []

            leaf_values = {}

            for node, path in leaf_nodes.values():

                # move scratch board from root to leaf
                for path_node in path[1:]:
                    scratch_board.push(path_node.move)

                # check terminal position
                if scratch_board.is_game_over(claim_draw = True):

                    node.expanded = True
                    
                    value = terminal_value(scratch_board)
        
                    node.value_prediction = value

                    leaf_values[id(node)] = value

                else:

                    batch_nodes.append(node)

                    # encode the position before undoing moves
                    encoded_positions.append(encode_board(scratch_board).squeeze(0))

                    # generate legal moves
                    legal_moves = list(scratch_board.legal_moves)

                    legal_moves_batch.append(legal_moves)

                    # generate policy indices
                    legal_ids_batch.append([
                        get_policy_index(scratch_board, move)
                        for move in legal_moves
                    ])

                # undo all moves to return to root
                for _ in path[1:]:
                    scratch_board.pop()

            # evaluate all non-terminal leaves together
            if batch_nodes:

                new_values = evaluate_and_expand_encoded_batch(
                    nodes = batch_nodes,
                    encoded_positions = encoded_positions,
                    legal_moves_batch = legal_moves_batch,
                    legal_ids_batch = legal_ids_batch,
                    model = self.model,
                    device = self.device
                )

                leaf_values.update(new_values)

            # backpropagation

            for path in paths:

                leaf = path[-1]

                leaf_value = leaf_values[id(leaf)]

                # remove temp penalty
                revert_virtual_loss(path, self.virtual_loss)

                backpropagate(path, leaf_value)

        return root
    

    def get_policy(self, root: Node, temperature = 1.0):

        moves = list(root.children.keys())

        visits = torch.tensor(
            [
                root.children[move].visit_count
                for move in moves
            ],
            dtype = torch.float32
        )

        if not moves:
            return {}

        # deterministic
        if visits.sum() == 0:

            probs = torch.tensor(
                [
                    root.children[move].prior_value
                    for move in moves
                ],
                dtype = torch.float32
            )

            probs = probs / probs.sum()

        elif temperature == 0:
            probs = torch.zeros_like(visits)

            probs[torch.argmax(visits)] = 1.0

        else: 
            # probability

            # calculate log visit counts
            log_visits = torch.full_like(
                visits,
                float('-inf')
            )

            visited = visits > 0

            log_visits[visited] = torch.log(
                visits[visited]
            )

            # apply temeprature
            probs = F.softmax(
                log_visits / temperature,
                dim = 0
            )

        return { 
            move: float(prob)
            for move, prob in zip(moves, probs.tolist())    
        }

    # select move
    def select_move(self, root: Node, temperature = 0, min_frac = 0.3):

        policy = self.get_policy(root, temperature)

        max_visits = max(child.visit_count for child in root.children.values())

        moves = [move for move in policy if root.children[move].visit_count >= min_frac * max_visits]

        probs = torch.tensor([policy[move] for move in moves], dtype = torch.float32)

        selected_index = torch.multinomial(probs, num_samples = 1).item()

        return moves[selected_index]

    def add_root_noise(self, root: Node, alpha = 0.3, epsilon = 0.25):

        children = list(root.children.values())

        if len(children) <= 1:
            return

        # generate dirichlet noise
        concentration = torch.full(
            (len(children),),
            alpha,
            dtype = torch.float32,
            device = self.device
        )

        noise = torch.distributions.Dirichlet(
            concentration
        ).sample().cpu().tolist()

        # mix original priors with noise
        for child, noise_value in zip(children, noise):

            child.root_prior_value = ((1 - epsilon) * child.prior_value + epsilon * noise_value)


class ReplayBuffer:

    def __init__(self, capacity = 20000):

        self.capacity = capacity

        # fixed size circular buffer
        self.storage = [None] * capacity

        self.position = 0
        self.size = 0

    def __len__(self):

        return self.size

    # add new training positions
    def add_samples(self, samples):

        for sample in samples:

            # keep everything on cpu
            stored_sample = {
                'board' : sample['board'].detach().to(device = 'cpu', dtype = torch.float32).contiguous(),
                'legal_ids' : sample['legal_ids'].detach().to(device = 'cpu', dtype = torch.long).contiguous(),
                'policy_target' : sample['policy_target'].detach().to(device = 'cpu', dtype = torch.float32).contiguous(),
                'policy_weight' : float(sample.get('policy_weight', 1)),
                'value_target' : float(sample['value_target'])
            }

            # insert sample
            self.storage[self.position] = stored_sample

            # advance circular position
            self.position = (self.position + 1) % self.capacity

            self.size = min(self.size + 1, self.capacity)

    # randomly sample training positions
    def sample(self, num_samples):

        sample_count = min(num_samples, self.size)

        # sample without replacement
        indices = random.sample(range(self.size), sample_count)

        return [
            self.storage[i]
            for i in indices
        ]

    # save buffer state
    def state_dict(self):

        return {
            'capacity' : self.capacity,
            'storage' : self.storage,
            'position' : self.position,
            'size' : self.size
        }

    # restore buffer state
    def load_state_dict(self, state):

        old_capacity = state['capacity']
        old_storage = state['storage']
        old_position = state['position']
        old_size = state['size']

        # reconstruct samples from oldest to newest
        if old_size < old_capacity:

            ordered_samples = old_storage[:old_size]

        else:

            ordered_samples = [
                old_storage[
                    (old_position + i) % old_capacity
                ]
                for i in range(old_size)
            ]

        # reset current buffer
        self.storage = [None] * self.capacity
        self.position = 0
        self.size = 0

        # keep newest samples that fit
        self.add_samples(ordered_samples[-self.capacity:])


# chess endgame
def syzygy_value(wdl):
    if wdl == 2:
        return 1
    elif wdl == -2:
        return -1
    else:
        return 0


def make_endgame_sample(board, policy, tablebase):

    if len(board.piece_map()) > 5:
        return None

    if board.castling_rights:
        return None

    if board.is_game_over(claim_draw = False):
        return None


    tb_board = board.copy(stack = False)
    tb_board.halfmove_clock = 0

    wdl = tablebase.get_wdl(tb_board)

    if wdl is None:
        return None

    legal_moves = list(tb_board.legal_moves)

    if not legal_moves:
        return None

    policy_target = torch.tensor(
        [
            policy.get(move, 0)
            for move in legal_moves
        ],
        dtype = torch.float32
    )

    target_sum = policy_target.sum()

    if target_sum <= 0:
        return None

    policy_target /= target_sum

    return {
        'board' : encode_board(tb_board).squeeze(0).detach().cpu(),
        'legal_ids' : torch.tensor(
            [
                get_policy_index(tb_board, move)
                for move in legal_moves
            ],
            dtype = torch.long
        ),
        'policy_target' : policy_target,
        'policy_weight' : 1,
        'value_target' : syzygy_value(wdl)
    }