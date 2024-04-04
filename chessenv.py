
import evaluate
import chess
import numpy
from typing import Union
import math

class chessEnv:
    def __init__(self, board: chess.Board) -> None:
        self.board = board
    
    def reset(self) -> Union[chess.Board, numpy.ndarray]:
        self.board.reset()
        return self.get_bitboard(self.board)
    
    def step(self, move: chess.Move) -> Union[chess.Board, bool]:
        eval = evaluate.move_value(self.board, move, False)
        self.board.push(move=move)
        print(self.board)
        # self.board.pop()
        if self.board.is_checkmate():
            print("\n\n\nCheckmate for ", "white\n\n\n" if self.board.turn else "black\n\n\n")
            eval = 1000000000000000 if self.board.turn else -1000000000000000
        if self.board.is_stalemate():
            print("\n\n\nStalemate\n\n\n")
            eval = 0
        if self.board.is_insufficient_material():
            print("\n\n\nInsufficient Material\n\n\n")
            eval = 0
        if self.board.is_fivefold_repetition():
            print("\n\n\nFivefold Repetition\n\n\n")
            eval = 0
        if self.board.status()!=chess.Status.VALID:
            print("\n\n\nInvalid\n\n\n")
        return self.get_bitboard(self.board), eval, (self.board.is_checkmate() or self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_fivefold_repetition() or self.board.status()!=chess.Status.VALID) , self.board.status() == chess.Status.VALID
    
    def get_board(self) -> chess.Board:
        return self.board


    def encode_move(self, move: chess.Move, index_only:bool, white_to_move: bool) -> numpy.ndarray:
    #     We represent the policy π(a|s) by a 8 × 8 × 76
    # stack of planes encoding a probability distribution over 4,672 possible moves. Each of the 8 × 8
    # positions identifies the square from which to “pick up” a piece. The first 56 planes encode
    # possible ‘queen moves’ for any piece: a number of squares [1..7] in which the piece will be
    # moved, along one of eight relative compass directions {N, N E, E, SE, S, SW, W, N W }
        letters_arr = ['a','b','c','d','e','f','g','h']
        from_uci_str = chess.square_name(move.from_square)
        from_uci_row = letters_arr.index(from_uci_str[0])
        from_uci_col = int(from_uci_str[1])-1
        to_uci_str = chess.square_name(move.to_square)
        to_uci_row = letters_arr.index(to_uci_str[0])
        to_uci_col = int(to_uci_str[1])-1

        # if white_to_move:
        #     from_uci_col = 7-from_uci_col
        #     from_uci_row = 7-from_uci_row
        #     to_uci_col = 7-to_uci_col
        #     to_uci_row = 7-to_uci_row

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
            dir = 2 if (up and not nu and nr) \
                else 1 if (up and not nu and right and not nr) \
                else 0 if (nu and right and not nr) \
                else 7 if (not up and not nu and right and not nr) \
                else 6 if (not up and not nu and nr) \
                else 5 if (not up and not nu and not right and not nr) \
                else 4 if (nu and not right and not nr) \
                else 3
            idx = (distance + 7*dir)-1
        elif promotion_uci_str is None: #knight move
            mp = [(2, 1), (2, -1), (-2, 1),(-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
            up_distance = to_uci_row - from_uci_row
            right_distance = to_uci_col - from_uci_col
            idx = 56
            idx = idx + mp.index((up_distance, right_distance))
            
            pass
        else: # promotion
            idx=64
            right_distance =  from_uci_row - to_uci_row
            mult = 0 if right_distance<0 else 1 if right_distance>0 else 2
            add = 1 if promotion_uci_str=="rook" else 2 if promotion_uci_str=="knight" else 3 if promotion_uci_str=="bishop" else 4
            idx = idx + ((add + 4*mult)-1)
            pass
        

        if(index_only):
            return None, (idx, from_uci_row, from_uci_col)
        arr = numpy.zeros(shape=[76, 8, 8])
        arr[idx][from_uci_row][from_uci_col] = 1
        return arr, (idx, from_uci_row, from_uci_col)

    def decode_move(self, array: numpy.ndarray, white_to_move: bool) -> chess.Move:
        from_col = numpy.where(array != 0)[1][0] + 1
        from_row = numpy.where(array != 0)[2][0] + 1
        idx = numpy.where(array != 0)[0][0]
        to_row = 0
        to_col = 0
        promo = None
        
        # if not white_to_move:
        #     from_col = 7-from_col
        #     from_row = 7-from_row
            

        if(idx<56):
            nb = idx
            distance = (nb % 7) + 1
            dir = math.floor(nb / 7)
            # {N, N E, E, SE, S, SW, W, N W }
            if dir==0 or dir==1:
                to_row = from_row + distance
                to_col = from_col if dir==0 else from_col + distance
            if dir==2 or dir==3:
                to_row = from_row if dir==2 else from_row-distance
                to_col = from_col + distance
            if dir==4 or dir==5:
                to_row = from_row - distance
                to_col = from_col if dir==4 else from_col - distance
            if dir==6 or dir==7:
                to_row = from_row if dir==6 else from_row+distance
                to_col = from_col - distance
            
        elif(idx>=56 and idx<64):
            nb = idx-56
            mp = [(2, 1), (2, -1), (-2, 1),(-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
            pair = mp[nb]
            up_distance = pair[0]
            right_distance = pair[1]
            to_row = from_row+right_distance
            to_col = from_col+up_distance
            
        else:
            #promo
            nb = idx-64
            mult = math.floor(nb/4) - 1
            add = nb % 4
            to_row = 8
            to_col = from_col-1 if mult==0 else from_col if mult==1 else from_col+1
            promo = chess.ROOK if add==0 else chess.KNIGHT if add==1 else chess.BISHOP if add==2 else chess.QUEEN
            
            #queen move, no promo
      
        return chess.Move(from_square=((from_col-1)+8*(from_row-1)), to_square=((to_col-1)+8*(to_row-1)), promotion=promo, drop=None)

    def get_bitboard(self, board: chess.Board) -> numpy.ndarray:
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
            
            # flip board according to turn
            if s==s.lower() and board.turn==chess.WHITE:
                n = n + 6
            elif s==s.upper() and board.turn==chess.BLACK:
                n = n + 6
            return n
        
        bitboard = board.piece_map()
        
        b_arr = numpy.zeros(shape=[12, 8, 8])
        for n in range(8):
            for m in range(8):
                if(bitboard.get(n + 8*m) is not None):
                    b_arr[map_symbol(bitboard[n + 8*m].symbol())][m][n] = 1

        
        return b_arr

if __name__ == '__main__':
    board = chess.Board()
    c = chessEnv(board)
    c.get_bitboard(board)

    board.push_san("e4")
    c.get_bitboard(board)