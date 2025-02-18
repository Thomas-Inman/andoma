import sys
import chess
import argparse
from movegeneration import next_move, next_move_agent
from agent import DeepQLearning
import os
import chessenv

env = None
dql = None

def talk():
    """
    The main input/output loop.
    This implements a slice of the UCI protocol.
    """
    board = chess.Board()
    depth = get_depth()

    while True:
        msg = input()
        command(depth, board, msg)


def command(depth: int, board: chess.Board, msg: str):
    """
    Accept UCI commands and respond.
    The board state is also updated.
    """
    msg = msg.strip()
    tokens = msg.split(" ")
    while "" in tokens:
        tokens.remove("")

    if msg == "quit":
        sys.exit()

    if msg == "uci":
        print("id name Andoma")  # Andrew/Roma -> And/oma
        print("id author Andrew Healey & Roma Parramore")
        print("uciok")
        return

    if msg == "isready":
        print("readyok")
        return

    if msg == "ucinewgame":
        return

    if msg.startswith("position"):
        if len(tokens) < 2:
            return

        # Set starting position
        if tokens[1] == "startpos":
            board.reset()
            moves_start = 2
        elif tokens[1] == "fen":
            fen = " ".join(tokens[2:8])
            board.set_fen(fen)
            moves_start = 8
        else:
            return

        # Apply moves
        if len(tokens) <= moves_start or tokens[moves_start] != "moves":
            return

        for move in tokens[(moves_start+1):]:
            board.push_uci(move)

    if msg == "d":
        # Non-standard command, but supported by Stockfish and helps debugging
        print(board)
        print(board.fen())
    
    if env is None:
        env = chessenv.chessEnv(board)

    if dql is None:
        dql = DeepQLearning(env, (12, 8, 8), 500, 64, 0.7, 0.9, 0.1, 0.95, False)
    if os.name == 'nt':
        dql.load("checkpoints\\model500.h5", "checkpoints\\targetModel500.h5")
    else:
        dql.load("checkpoints/model500.h5", "checkpoints/targetModel500.h5")

    if msg[0:2] == "go":
        _move = next_move_agent(depth, board, dql)
        print(f"bestmove {_move}")
        return


def get_depth() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", default=3, help="provide an integer (default: 3)")
    args = parser.parse_args()
    return max([1, int(args.depth)])
