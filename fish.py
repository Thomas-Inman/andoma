import chess
from stockfish import Stockfish

class Fish:
    def __init__(self, elo=200, skill=2):
        self.fish = Stockfish(depth=1, parameters={"UCI_Elo": elo, "Skill Level": skill, "UCI_Chess960": "false"})
    
    def get_move(self, fen):
        # Given a FEN return the best move
        assert self.fish.is_fen_valid(fen)
        self.fish.set_fen_position(fen)
        return self.fish.get_best_move_time(100)
        

if __name__ == "__main__":
    board = chess.Board()
    goldFish = Fish()
    print(board.fen())
    print(board.push_san(goldFish.get_move(board.fen())))