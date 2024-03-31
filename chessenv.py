
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
#     We represent the policy π(a|s) by a 8 × 8 × 73
# stack of planes encoding a probability distribution over 4,672 possible moves. Each of the 8 × 8
# positions identifies the square from which to “pick up” a piece. The first 56 planes encode
# possible ‘queen moves’ for any piece: a number of squares [1..7] in which the piece will be
# moved, along one of eight relative compass directions {N, N E, E, SE, S, SW, W, N W }
    letters_arr = ['a','b','c','d','e','f','g','h']
    from_uci_str = chess.square_name(move.from_square())
    from_uci_row = letters_arr.index(from_uci_str[0])
    from_uci_col = int(from_uci_str[1])-1
    to_uci_str = chess.square_name(move.to_square())
    to_uci_row = letters_arr.index(to_uci_str[0])
    to_uci_col = int(to_uci_str[1])-1
    queen_move = (abs(from_uci_col-to_uci_col)==abs(from_uci_row-to_uci_row)) or ((from_uci_col==to_uci_col) or (from_uci_row==to_uci_row))
    idx = 0
    promotion_uci_str = chess.piece_name(move.promotion) if move.promotion is not None else None
    if queen_move and promotion_uci_str is None:
        distance = max(abs(from_uci_col-to_uci_col), abs(from_uci_row-to_uci_row))
        right = to_uci_col>from_uci_col
        nr = to_uci_col==from_uci_col
        up = to_uci_row>from_uci_row
        nu = to_uci_row==from_uci_row
        # {N, N E, E, SE, S, SW, W, N W }
        dir = 0 if (up and not nu and nr) \
            else 1 if (up and not nu and right and not nr) \
            else 2 if (nu and right and not nr) \
            else 3 if (not up and not nu and right and not nr) \
            else 4 if (not up and not nu and nr) \
            else 5 if (not up and not nu and not right and not nr) \
            else 6 if (nu and not right and not nr) \
            else 7
        idx = (distance + 7*dir)-1
    elif promotion_uci_str is None: #knight move
        up_distance = to_uci_row - from_uci_row
        right_distance = to_uci_col - from_uci_col
        idx = 56
        idx = idx + (0 if up_distance==2 and right_distance==1\
                     else 1 if up_distance==2 and right_distance==-1\
                        else 2 if up_distance==-2 and right_distance==1\
                            else 3 if up_distance==-2 and right_distance==-1\
                                else 4 if up_distance==1 and right_distance==2\
                                    else 5 if up_distance==1 and right_distance==-2\
                                        else 6 if up_distance==-1 and right_distance==2\
                                            else 7)
        
        pass
    else: # promotion
        idx=64
        right_distance = to_uci_col - from_uci_col
        mult = 0 if right_distance<0 else 1 if right_distance==0 else 2
        add = 1 if promotion_uci_str=="rook" else 2 if promotion_uci_str=="knight" else 3 if promotion_uci_str=="bishop" else 4
        idx = idx + ((add + 4*mult)-1)
        pass
    
    arr = numpy.zeros(shape=[8, 8, 76])
    arr[from_uci_col][from_uci_row][idx] = 1
    

    return arr

def decode_move(array: numpy.ndarray) -> chess.Move:
    from_col = numpy.where(array != 0)[0][0]
    from_row = numpy.where(array != 0)[1][0]
    idx = numpy.where(array != 0)[2][0]
    to_row = 0
    to_col = 0
    promo = None
    if(idx<64):
        to_row = 
        pass
        #queen move, no promo
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

