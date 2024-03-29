
import evaluate
import chess
import numpy
from typing import Union

class chessEnv:
    def __init__(self, board: chess.Board) -> None:
        self.board = board
    
    def reset_env(self) -> Union[chess.Board, numpy.ndarray]:
        self.board.reset()
        return self.board
    
    def make_move(self, board: chess.Board):
        board.push(move=chess.Move())
    


def get_bitboard(board: chess.Board) -> numpy.ndarray:
    def map_symbol(s: str) -> int:
        # n = 0
        if s.upper()=='P':
            n=0
        elif s.upper()=='N':
            n=1
        elif s.upper()=='B':
            n=2
        elif s.upper()=='R':
            n=3
        elif s.upper()=='Q':
            n=4
        else:
            n=5

        if s==s.lower():
            n = n + 6
        return n
    
    bitboard = board.piece_map()
    
    b_arr = numpy.zeros(shape=[64, 12])
    for n in range(64):
        if(bitboard.get(n) is not None):
            b_arr[n][map_symbol(bitboard[n].symbol())] = 1
    print(b_arr)
    
    return bitboard

