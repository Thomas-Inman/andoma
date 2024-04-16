import chess
from stockfish import Stockfish
from pathlib import Path

class Fish:
    def __init__(self, elo=200, skill=2):
        # self.fish = Stockfish(path=str(Path("C:\\Users\\marie\\Downloads\\stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe")),depth=1, parameters={"UCI_Elo": elo, "Skill Level": skill, "UCI_Chess960": "false"})
        self.fish = Stockfish(path="/home/marie-ezra/Downloads/stockfish/stockfish-ubuntu-x86-64-avx2",depth=1, parameters={"UCI_Elo": elo, "Skill Level": skill, "UCI_Chess960": "false"})
    
    def get_move(self, fen):
        # Given a FEN return the best move
        assert self.fish.is_fen_valid(fen)
        self.fish.set_fen_position(fen)
        move = self.fish.get_best_move_time(100)
        return move
        

if __name__ == "__main__":
    board = chess.Board()
    goldFish = Fish()
    print(board.fen())
    print(board.push_san(goldFish.get_move(board.fen())))