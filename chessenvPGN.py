import chessenv
import chess
from chess import pgn
import numpy
from typing import Union
import math
import evaluate

class chessEnvPGN(chessenv.chessEnv):
    def __init__(self, board: chess.Board, PGN_archive_path: str = None) -> None:
        self.board = board
        self.PGN_archive_path = PGN_archive_path
        self.PGN_index = 0
        assert self.PGN_archive_path is not None
        self.PGN_file = open(self.PGN_archive_path)
        self.game = None
        self.current_index_ingame = 0
        # read number of games in PGN directory

    def step(self, move: chess.Move) -> Union[chess.Board, bool]:
        eval = evaluate.move_value(self.board, move, False)
        self.board.push(move=move)
        print(self.board)
        # self.board.pop()
        if self.board.is_checkmate():
            print("\n\n\nCheckmate for ", "white\n\n\n" if self.board.turn else "black\n\n\n")
            eval = 10000 if self.board.turn else -10000
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
        return chessenv.get_bitboard(self.board), eval, (self.board.is_checkmate() or self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_fivefold_repetition() or self.board.status()!=chess.Status.VALID) , self.board.status() == chess.Status.VALID
        

    def reset(self) -> Union[chess.Board, numpy.ndarray]:
        self.board.reset()
        # read next game from PGN archive
        self.game = pgn.read_game(self.PGN_file)
        return chessenv.get_bitboard(self.board)

    def get_board(self) -> chess.Board:
        return self.board
    
    def next_move(self, depth: int, board: chess.Board, training=False, agent=None, epsilon=0, episode=0):
        move = self.game.mainline_moves()[self.current_index_ingame]
        self.current_index_ingame += 1
        return move
    