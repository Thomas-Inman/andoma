# inspired by the https://github.com/thomasahle/sunfish user inferface

import chess
import argparse
from movegeneration import next_move, next_move_agent
import chessenv
from agent import DeepQLearning
import os


def start():
    """
    Start the command line user interface.
    """
    board = chess.Board()
    user_side = (
        chess.WHITE if input("Start as [w]hite or [b]lack:\n") == "w" else chess.BLACK
    )

    if user_side == chess.WHITE:
        print(render(board))
        board.push(get_move(board))

    while not board.is_game_over():
        board.push(next_move(get_depth(), board, debug=False))
        print(render(board))
        board.push(get_move(board))

    print(f"\nResult: [w] {board.result()} [b]")


def start_with_agent():
    """
    Start the command line user interface.
    """
    board = chess.Board()
    user_side = (
        chess.WHITE if input("Start as [w]hite or [b]lack:\n") == "w" else chess.BLACK
    )
    env = chessenv.chessEnv(chess.Board())
    dql = DeepQLearning(env, (12, 8, 8), 500, 64, 0.7, 0.9, 0.1, 0.95, False)
    if os.name == 'nt':
        dql.load("checkpoints\\model500.h5", "checkpoints\\targetModel500.h5")
    else:
        dql.load("checkpoints/model500.h5", "checkpoints/targetModel500.h5")

    if user_side == chess.WHITE:
        print(render(board))
        board.push(get_move(board))
        # board.push(next_move(1, board, debug=False))

    while not board.is_game_over():
        board.push(next_move_agent(2, board, debug=False, agent=dql))
        print(render(board))
        try:
            board.push(get_move(board))
        except:
            break
        # try:
        #     board.push(next_move(1, board, debug=False))
        # except:
        #     break

    print(f"\nResult: [w] {board.result()} [b]")


def render(board: chess.Board) -> str:
    """
    Print a side-relative chess board with special chess characters.
    """
    board_string = list(str(board))
    uni_pieces = {
        "R": "♖",
        "N": "♘",
        "B": "♗",
        "Q": "♕",
        "K": "♔",
        "P": "♙",
        "r": "♜",
        "n": "♞",
        "b": "♝",
        "q": "♛",
        "k": "♚",
        "p": "♟",
        ".": "·",
    }
    for idx, char in enumerate(board_string):
        if char in uni_pieces:
            board_string[idx] = uni_pieces[char]
    ranks = ["1", "2", "3", "4", "5", "6", "7", "8"]
    display = []
    for rank in "".join(board_string).split("\n"):
        display.append(f"  {ranks.pop()} {rank}")
    if board.turn == chess.BLACK:
        display.reverse()
    display.append("    a b c d e f g h")
    return "\n" + "\n".join(display)


def get_move(board: chess.Board) -> chess.Move:
    """
    Try (and keep trying) to get a legal next move from the user.
    Play the move by mutating the game board.
    """
    move = input(f"\nYour move (e.g. {list(board.legal_moves)[0]}):\n")

    for legal_move in board.legal_moves:
        if move == str(legal_move):
            return legal_move
    return get_move(board)


def get_depth() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", default=3, help="provide an integer (default: 3)")
    args = parser.parse_args()
    return max([1, int(args.depth)])


if __name__ == "__main__":
    try:
        start_with_agent()
    except KeyboardInterrupt:
        pass
