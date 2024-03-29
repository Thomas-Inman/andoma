
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
    
    def make_move(self, move: chess.Move) -> Union[chess.Board, bool]:
        self.board.push(move=move)
        if(self.board.status() == chess.Status.VALID):
            return self.board, True
        self.board.pop()
        return self.board, False

    


def encode_move(move: chess.Move) -> numpy.ndarray:
    uci_str = move.uci()
    letters_arr = ['a','b','c','d','e','f','g','h']
    from_uci_str = chess.square_name(move.from_square())
    from_uci_row = letters_arr.index(from_uci_str[0])
    from_uci_col = int(from_uci_str[1])
    
    to_uci_str = chess.square_name(move.to_square())
    promotion_uci_str = chess.piece_name(move.promotion) if move.promotion is not None else None
    arr = numpy.zeros(shape=[8, 8, 73])
    

    return

def get_bitboard(board: chess.Board) -> numpy.ndarray:
    def map_symbol(s: str) -> int:
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

