from pathlib import Path
import torch
import chess
import chess.pgn

def save_model(model_state_dict, optimiser_state_dict, scheduler_state_dict, game_count, model_path):
    temp_path = model_path.with_suffix('.tmp')
    
    torch.save({
        'model_state_dict' : model_state_dict,
        'optimiser_state_dict' : optimiser_state_dict,
        'schedular_state_dict' : scheduler_state_dict,
        'game_count' : game_count
    }, temp_path)

    temp_path.replace(model_path)

    print('model saved')



def save_replay(replay, replay_path):
    temp_path = replay_path.with_suffix('.tmp')

    torch.save(replay.state_dict(), temp_path)

    temp_path.replace(replay_path)

    print(f'replay saved ({len(replay)})')



def save_pgn(board, game_count, bot_play_white, pgn_path):
     # export pgn
    pgn_game = chess.pgn.Game()

    # headers
    pgn_game.headers['Event'] = 'stockcopy v4 vs stockcopy v4'
    pgn_game.headers['Site'] = 'local'
    pgn_game.headers['Date'] = '????.??.??'
    pgn_game.headers['Round'] = str(game_count)

    pgn_game.headers['White'] = 'stockcopy v5' if bot_play_white else 'stockfish'
    pgn_game.headers['Black'] = 'stockcopy v5' if not bot_play_white else 'stockfish'

    pgn_game.headers['Result'] = board.result(claim_draw = True)

    # add moves
    node = pgn_game

    for move in board.move_stack:
        node = node.add_main_variation(move)

    # export to pgn txt
    exporter = chess.pgn.StringExporter(
        headers = True,
        variations = False,
        comments = True
    )

    pgn_text = pgn_game.accept(exporter)

    # save to file
    pgn_file_path = pgn_path / f'game_{game_count:05d}.pgn'

    with open(pgn_file_path, 'w', encoding = 'utf-8') as file:
        file.write(pgn_text)

    print('pgn saved')



def save_checkpoint(model_state_dict, optimiser_state_dict, scheduler_state_dict, game_count):

    save_checkpoint_file = Path(f'C:/Users/sampl/Documents/code/personal/stockcopy/src/stockcopy/model/checkpoint_{game_count}.pt')

    torch.save({
        'model_state_dict' : model_state_dict,
        'optimiser_state_dict' : optimiser_state_dict,
        'schedular_state_dict' : scheduler_state_dict,
        'game_count' : game_count
    }, save_checkpoint_file)

    print('checkpoint saved')