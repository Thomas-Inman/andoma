# inspired by the https://github.com/thomasahle/sunfish user inferface

import chess
import argparse
from movegeneration import next_move, next_move_agent
import chessenv
from agent import DeepQLearning
import os
import numpy as np
from fish import Fish
from pathlib import Path

# get working dir using pathlib
PATH = str(Path(__file__).parent.absolute())


def runGame(board, dql, Fish):
    """
    Start the command line user interface.
    """

    
    # Init DQL with model params

    
    # Select opponent 
    if np.random.default_rng().random() < 0.5:
        fishSide = chess.WHITE
    else:
        fishSide = chess.BLACK


    if fishSide == chess.WHITE:
        #print(render(board))
        board.push_san(get_move(board, Fish))
        print(board)
        # board.push(next_move(1, board, debug=False))
    i = 0
    while not board.is_game_over():
        print(i)
        i+= 1
        board.push(next_move_agent(2, board, debug=False, agent=dql))
        print(board)
        if board.is_game_over():
            break
        try:
            board.push_san(get_move(board, Fish))
        except:
            print("loop broken")
            break

    return fishSide, board.result()



def get_move(board: chess.Board, Fish) -> chess.Move:
    """
    Try (and keep trying) to get a legal next move from the stockfish.
    Play the move by mutating the game board.
    """

    return Fish.get_move(board.fen())


def get_depth() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", default=3, help="provide an integer (default: 3)")
    args = parser.parse_args()
    return max([1, int(args.depth)])


if __name__ == "__main__":
        # Initialize game board and env
    # board = chess.Board()
    # board.reset()
    # env = chessenv.chessEnv(chess.Board())
    
    # # Load model
    # if os.name == 'nt':
    #     modelName, targetModelName = f"checkpoints\\3_model400.h5", f"checkpoints\\3_targetModel400.h5"
    # else:
    #     modelName, targetModelName = "checkpoints/3_model400.h5", "checkpoints/3_targetModel400.h5"
    # # init dqn
    # dql = DeepQLearning(env, (12, 8, 8), 500, 64, 0.7, 0.9, 0.1, 0.95, False, True)
    # if os.name == 'nt':
    #     dql.load(modelName, targetModelName)
    #     try:
    #         f = open(PATH + f"\\results\\3_400_res.csv", 'a')
    #     except FileNotFoundError:
    #         f = open(PATH + f"\\results\\3_400_res.csv", 'w')
    # else:
    #     dql.load(PATH + modelName, PATH + targetModelName)
    #     try:
    #         f = open(PATH + f"/results/3_400_res.csv", 'a')
    #     except FileNotFoundError:
    #         f = open(PATH + f"/results/3_400_res.csv", 'w')
    # for i in range(20):
    #     board.reset()
        
    #     #f.write("fishStart, w, b\n")
    #     try:
    #         fish = Fish()
    #         fishStart, res = runGame(board, dql, fish)
    #         print("game: " + str(i) + " " + str(fishStart) + str(res))
    #         f.write(str(fishStart) + "," + str(res[0]) + "," + str(res[-1]) + '\n')
    #     except KeyboardInterrupt:
    #         f.close()
    #         break
    # f.close()


        # Initialize game board and env
    board = chess.Board()
    board.reset()
    env = chessenv.chessEnv(chess.Board())
    
    # Load model
    if os.name == 'nt':
        modelName, targetModelName = f"checkpoints\\5_model500.h5", f"checkpoints\\5_targetModel500.h5"
    else:
        modelName, targetModelName = "checkpoints/3_model400.h5", "checkpoints/3_targetModel400.h5"
    # init dqn
    dql = DeepQLearning(env, (12, 8, 8), 500, 64, 0.7, 0.9, 0.1, 0.95, False, True)
    if os.name == 'nt':
        dql.load(modelName, targetModelName)
        try:
            f = open(PATH + f"\\results\\5_500_res.csv", 'a')
        except FileNotFoundError:
            f = open(PATH + f"\\results\\5_500_res.csv", 'w')
    else:
        dql.load(PATH + modelName, PATH + targetModelName)
        try:
            f = open(PATH + f"/results/4_300_res.csv", 'a')
        except FileNotFoundError:
            f = open(PATH + f"/results/4_300_res.csv", 'w')
    for i in range(20):
        board.reset()
        
        #f.write("fishStart, w, b\n")
        try:
            fish = Fish()
            fishStart, res = runGame(board, dql, fish)
            print("game: " + str(i) + " " + str(fishStart) + str(res))
            f.write(str(fishStart) + "," + str(res[0]) + "," + str(res[-1]) + '\n')
        except KeyboardInterrupt:
            f.close()
            break
    f.close()

            # Initialize game board and env
    board = chess.Board()
    board.reset()
    env = chessenv.chessEnv(chess.Board())
    
    # Load model
    if os.name == 'nt':
        modelName, targetModelName = f"checkpoints\\2_model500.h5", f"checkpoints\\2_targetModel500.h5"
    else:
        modelName, targetModelName = "checkpoints/3_model400.h5", "checkpoints/3_targetModel400.h5"
    # init dqn
    dql = DeepQLearning(env, (12, 8, 8), 500, 64, 0.7, 0.9, 0.1, 0.95, False, False)
    if os.name == 'nt':
        dql.load(modelName, targetModelName)
        try:
            f = open(PATH + f"\\results\\2_500_res.csv", 'a')
        except FileNotFoundError:
            f = open(PATH + f"\\results\\2_500_res.csv", 'w')
    else:
        dql.load(PATH + modelName, PATH + targetModelName)
        try:
            f = open(PATH + f"/results/4_300_res.csv", 'a')
        except FileNotFoundError:
            f = open(PATH + f"/results/4_300_res.csv", 'w')
    for i in range(20):
        board.reset()
        
        #f.write("fishStart, w, b\n")
        try:
            fish = Fish()
            fishStart, res = runGame(board, dql, fish)
            print("game: " + str(i) + " " + str(fishStart) + str(res))
            f.write(str(fishStart) + "," + str(res[0]) + "," + str(res[-1]) + '\n')
        except KeyboardInterrupt:
            f.close()
            break
    f.close()
