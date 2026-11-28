import os
import requests
import time
import json
import chess
from dotenv import load_dotenv
from pathlib import Path
import torch
import threading

import nnmodel
import mcts
import buffer
import analyse
import training
import saving


# get lichess token
load_dotenv('.env')
lichess_token = os.getenv('LICHESS_TOKEN')


# load bot
model_path = Path(r'C:\Users\sampl\Documents\code\personal\stockcopy\src\stockcopy\model\test_model.pt')





class Ponderer:

    def __init__(self, searcher):

        self.searcher = searcher
        self.thread = None
        self.stop_event = None

        self.root = None


    def start(self, board, root):

        # stop if pondering session
        if self.thread is not None:
            self.stop()

        self.root = root

        self.stop_event = threading.Event()

        board_copy = board.copy(stack = True)

        def worker():

            try:

                print('started pondering')

                self.root = self.searcher.search(
                    board = board_copy,
                    num_simulations = None,
                    root = self.root,
                    use_root_noise = False,
                    stop_event = self.stop_event
                )

                print('pondering done')
    
            except Exception as e:
                print('error: ', e)
                self.stop_event.set()

        self.thread = threading.Thread(
            target = worker,
            daemon = True
        )

        self.thread.start()

    def stop(self):

        # check if there is a thread
        if self.thread is None:
            return self.root

        # set to stop
        self.stop_event.set()

        # wait till batch finishes
        self.thread.join()

        self.thread = None

        return self.root





# create match

def create_match():

    session = requests.Session()

    session.headers.update({
        'Authorization' : f'Bearer {lichess_token}',
        'User-Agent' : 'stockcopy the doom and despair'
    })

    # create open challenge
    response = requests.post(
        'https://lichess.org/api/challenge/open',
        data = {
            'rated' : 'false',
            'name' : 'play against the doom and despair'
        },
        headers = {
            'User-Agent' : 'stockcopy the doom and despair'
        },
        timeout = 30
    )

    response.raise_for_status()
    result = response.json()

    # get challenge info
    challenge = result.get('challenge', result)
    challenge_id = challenge['id']

    # bot joins
    response = session.post(
        f'https://lichess.org/api/challenge/{challenge_id}/accept',
        timeout = 30
    )

    response.raise_for_status()

    # get link
    link = result.get('url')

    print(link)

    play_match(challenge_id, session)

    return link

# make move
def play_move(board, game_id, session):

    global root

    # run mcts
    root = searcher.search(
        board = board,
        num_simulations = 6400,
        root = root,
        use_root_noise = False
    )

    move = searcher.select_move(
        root = root,
        temperature = 0
    )

    next_root = root.children[move]

    # play move

    root = next_root

    uci_move = move.uci()
    
    move_response = session.post(
        f'https://lichess.org/api/bot/game/{game_id}/move/{uci_move}',
        timeout = 30
    )

    move_response.raise_for_status()

    board_copy = board.copy(stack = True)

    board_copy.push(move)

    return board_copy




# play match

def play_match(game_id, session):

    global root

    # wait for game to start
    while True:

        response = session.get(
            f'https://lichess.org/api/bot/game/stream/{game_id}',
            stream = True,
            timeout = (10, None)
        )

        # if game hasnt started
        if response.status_code == 404:
            response.close()
            time.sleep(3)
            continue

        response.raise_for_status()
        break

    print('game started')

    with response:

        board = chess.Board()
        board_copy = chess.Board()

        for line in response.iter_lines():

            # skip if nothing
            if not line:
                continue

            # grab event
            event = json.loads(line)

            print(event)

            type = event['type']


            
            # check if game started
            if type == 'gameFull':

                # check if bot is black or white
                white = event['white']

                bot_is_white = white.get('id') == 'stockcopy'

                if bot_is_white:
                    board_copy = play_move(board, game_id, session)
                else:
                    ponderer.start(board_copy, root)
                



            # check if a move is made
            if type == 'gameState':

                moves = event['moves'].split()

                board.push_uci(moves[-1])

                # check if its the bot move
                if (len(moves) % 2 == 0) == bot_is_white:

                    root = ponderer.stop()
                    root = root.children.get(moves[-1]) if root is not None else None
                    board_copy = play_move(board, game_id, session)
        
                else:
                    print(board, board_copy)

                    ponderer.start(board_copy, root)
                    

            












model = nnmodel.StockCopy().to('cuda')
searcher = mcts.MCTS(model)
ponderer = Ponderer(searcher)

# load model
if model_path.exists():

    checkpoint = torch.load(
        model_path,
        map_location = 'cuda'
    )

    model.load_state_dict(checkpoint['model_state_dict'])

    game_count = checkpoint.get('game_count', 0)

    print('\nloaded model')

model.eval()

root = None

  


create_match()