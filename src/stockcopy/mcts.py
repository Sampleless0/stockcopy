from __future__ import annotations
import chess
import torch
import torch.nn.functional as F
import math

from . import encoder
from . import inference

# add node class
class Node:

    def __init__(self, move: chess.Move | None = None, initial_prob: float = 1):

        self.move = move
        self.initial_prob = initial_prob

        self.root_initial_prob = None

        self.visit_count = 0
        self.value_sum = 0

        self.children: dict[chess.Move, Node] = {}

        self.expanded = False
        self.value_prediction = None

    @property
    def q_value(self):

        # calculate q_value using average sum
        
        if self.visit_count == 0:
            return 0

        return self.value_sum / self.visit_count


    # adding child (branching)
    def add_child(self, move: chess.Move, initial_prob: float):

        # create new child
        child = Node(move, initial_prob)

        # assign child to self
        self.children[move] = child

        return child


    # select child
    def select_child(self, c_puct = 1.5, use_root_initial_prob = False):

        # get best score (best move sequence)
        best_score = float('-inf')
        best_move = None
        best_child = None

        # check amount of visits to itself
        parent_visits = self.visit_count

        sqrt_parent = math.sqrt(parent_visits + 1)

        # check each move
        for move, child in self.children.items():

            # use noisy prior only when selecting from current root
            if use_root_initial_prob and child.root_initial_prob is not None:
                initial_prob = child.root_initial_prob
            else:
                initial_prob = child.initial_prob


            # add first play urgency
            fpu_reduction = 0.25

            parent_value = self.value_prediction if self.value_prediction is not None else 0

            # exploitation
            q = child.q_value if child.visit_count > 0 else parent_value - fpu_reduction

            # exploration
            u = c_puct * initial_prob * sqrt_parent / (1 + child.visit_count)

            # calculate PUCT score
            score = q + u

            # find highest score
            if score > best_score:

                best_score = score
                best_move = move
                best_child = child

        return best_score, best_move, best_child



# mcts functions
def terminal_value(board: chess.Board):

    outcome = board.outcome(claim_draw = True)

    # draw
    if outcome.winner is None:
        return 0

    # win / loss
    return 1 if outcome.winner == board.turn else -1

def get_policy_index(move, perspective):

    plane, row, col = encoder.encode_move(move, perspective)
    
    return plane * 64 + row * 8 + col

def evaluate_and_expand_batch(nodes: list[Node], positions, legal_moves_batch, legal_indices_batch, model):

    values = {}

    if not nodes:
        return values

    input_batch = torch.stack(
        positions
    ).to('cuda')

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
    for i, indices in enumerate(legal_indices_batch):

        count = len(indices)

        legal_indices[i, :count] = torch.tensor(
            indices,
            dtype = torch.long
        )

        legal_mask[i, :count] = True

    # transfer to gpu
    legal_indices = legal_indices.to('cuda')
    legal_mask = legal_mask.to('cuda')


    # nn evaluation
    prior_batch, value_batch = inference.evaluate_policy_batch(
        model = model,
        input_batch = input_batch,
        legal_indices = legal_indices,
        legal_mask = legal_mask,
    )

    # expand each position
    for i, node in enumerate(nodes):

        legal_moves = legal_moves_batch[i]
        initial_probs = prior_batch[i][:len(legal_moves)]
                    
        for move, prior in zip(legal_moves, initial_probs):

            node.add_child(
                move = move,
                initial_prob = prior
            )

        node.expanded = True

        node.value_prediction = value_batch[i]

        values[id(node)] = value_batch[i]


    return values


# virtual loss
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


class MCTS:

    def __init__(self, model, c_puct = 1.5, batch_size = 8, virtual_loss = 1):
        self.model = model
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.virtual_loss = virtual_loss

    # search for best move
    def search(self, board: chess.Board, num_simulations = 800, root: Node = None, use_root_noise = False):
        self.model.eval()

        # create a root
        if root is None:
            root = Node()

        scratch_board = board.copy(stack = True)

        # initial expansion if necessary
        if not root.expanded:

            # check for game end
            if board.is_game_over(claim_draw = True):
                root.children.clear()
                root.expanded = True
                root.value_prediction = terminal_value(board)
                return root

            else:

                # encode position
                encoded_position = encoder.encode_board(board).squeeze(0)

                # get legal moves
                legal_moves = list(board.legal_moves)

                # get legal indices
                legal_indices = [get_policy_index(move, board.turn) for move in legal_moves]

                evaluate_and_expand_batch([root], [encoded_position], [legal_moves], [legal_indices], self.model)

        # reset temporary root initial probs
        for child in root.children.values():
            child.root_initial_prob = None

        # apply noise once per search
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
                
                    _, _, child = node.select_child(c_puct = self.c_puct, use_root_initial_prob = (node is root))
    
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
            legal_indices_batch = []

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
                    encoded_positions.append(encoder.encode_board(scratch_board).squeeze(0))

                    # generate legal moves
                    legal_moves = list(scratch_board.legal_moves)

                    legal_moves_batch.append(legal_moves)

                    # generate policy indices
                    legal_indices_batch.append([
                        get_policy_index(move, scratch_board.turn)
                        for move in legal_moves
                    ])

                # undo all moves to return to root
                for _ in path[1:]:
                    scratch_board.pop()

            # evaluate all non-terminal leaves together
            if batch_nodes:

                new_values = evaluate_and_expand_batch(
                    nodes = batch_nodes,
                    positions = encoded_positions,
                    legal_moves_batch = legal_moves_batch,
                    legal_indices_batch = legal_indices_batch,
                    model = self.model,
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
                    root.children[move].initial_prob
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
            device = 'cuda'
        )

        noise = torch.distributions.Dirichlet(
            concentration
        ).sample().cpu().tolist()

        # mix original priors with noise
        for child, noise_value in zip(children, noise):

            child.root_initial_prob = ((1 - epsilon) * child.initial_prob + epsilon * noise_value)

