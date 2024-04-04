from typing import Dict, List, Any
import chess
import time
from evaluate import evaluate_board, move_value, check_end_game
import numpy
import random
import os


debug_info: Dict[str, Any] = {}


MATE_SCORE     = 1000000000
MATE_THRESHOLD =  999000000


def next_move(depth: int, board: chess.Board, debug=True) -> chess.Move:
    """
    What is the next best move?
    """
    debug_info.clear()
    debug_info["nodes"] = 0
    t0 = time.time()

    move = minimax_root(depth, board)

    debug_info["time"] = time.time() - t0
    if debug == True:
        print(f"info {debug_info}")
    return move

def next_move_agent(depth: int, board: chess.Board, debug=True, agent=None) -> chess.Move:
    """
    What is the next best move?
    """
    debug_info.clear()
    debug_info["nodes"] = 0
    t0 = time.time()

    # use agent to get the next move
    state = agent.env.get_bitboard(board)
    state = numpy.reshape(state, [1, 12, 8, 8])
    action, _ = agent.act(state)
    move = agent.env.decode_move(action, agent.env.get_board().turn)
    # check if the move is legal
    if not board.is_legal(move):
        print("DEBUG: Illegal move, playing random move")
        move = agent.env.decode_move(agent.random_move(), agent.env.get_board().turn)
    debug_info["time"] = time.time() - t0
    if debug == True:
        print(f"info {debug_info}")
    return move




def get_ordered_moves(board: chess.Board) -> List[chess.Move]:
    """
    Get legal moves.
    Attempt to sort moves by best to worst.
    Use piece values (and positional gains/losses) to weight captures.
    """
    end_game = check_end_game(board)

    def orderer(move):
        return move_value(board, move, end_game)

    in_order = sorted(
        board.legal_moves, key=orderer, reverse=(board.turn == chess.WHITE)
    )
    return list(in_order)

def minimax_root(depth: int, board: chess.Board, training=False, agent=None) -> chess.Move:
    """
    What is the highest value move per our evaluation function?
    """
    # White always wants to maximize (and black to minimize)
    # the board score according to evaluate_board()
    maximize = board.turn == chess.WHITE
    best_move = -float("inf")
    if not maximize:
        best_move = float("inf")

    moves = get_ordered_moves(board)
    best_move_found = moves[0]

    for move in moves:
        board.push(move)
        # Checking if draw can be claimed at this level, because the threefold repetition check
        # can be expensive. This should help the bot avoid a draw if it's not favorable
        # https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.can_claim_draw
        if board.can_claim_draw():
            value = 0.0
        else:
            value = minimax(depth - 1, board, -float("inf"), float("inf"), not maximize, training, agent)
            # print(value)
        board.pop()
        if maximize and value >= best_move:
            best_move = value
            best_move_found = move
        elif not maximize and value <= best_move:
            best_move = value
            best_move_found = move

    return best_move_found

def minimax_root_with_value(depth: int, board: chess.Board, training=False, agent=None, epsilon=0) -> chess.Move:
    """
    What is the highest value move per our evaluation function?
    """
    # White always wants to maximize (and black to minimize)
    # the board score according to evaluate_board()
    maximize = board.turn == chess.WHITE
    best_move = -float("inf")
    if not maximize:
        best_move = float("inf")

    moves = get_ordered_moves(board)
    best_move_found = moves[0]

    for move in moves:
        board.push(move)
        # Checking if draw can be claimed at this level, because the threefold repetition check
        # can be expensive. This should help the bot avoid a draw if it's not favorable
        # https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.can_claim_draw
        if board.can_claim_draw():
            value = 0.0
        else:
            value = minimax(depth - 1, board, -float("inf"), float("inf"), not maximize, training, agent, epsilon)
            # print(value)
        board.pop()
        if maximize and value >= best_move:
            best_move = value
            best_move_found = move
        elif not maximize and value <= best_move:
            best_move = value
            best_move_found = move

    return best_move_found, best_move



def minimax(
    depth: int,
    board: chess.Board,
    alpha: float,
    beta: float,
    is_maximising_player: bool,
    training=False,
    agent=None,
    epsilon=0
) -> float:
    """
    Core minimax logic.
    https://en.wikipedia.org/wiki/Minimax
    """
    debug_info.clear()
    debug_info["nodes"] = 0
    t0 = time.time()
    debug_info["nodes"] += 1

    if board.is_checkmate():
        # The previous move resulted in checkmate
        return -MATE_SCORE if is_maximising_player else MATE_SCORE
    # When the game is over and it's not a checkmate it's a draw
    # In this case, don't evaluate. Just return a neutral result: zero
    elif board.is_game_over():
        return 0

    if depth == 0:
        if numpy.random.rand() <= epsilon:
            return evaluate_board(board)
        else:
            return evaluate_board(board) if (agent is None or not training) else agent.evaluate_board(board)
        

    if is_maximising_player:
        best_move = -float("inf")
        moves = get_ordered_moves(board)
        for move in moves:
            board.push(move)
            curr_move = minimax(depth - 1, board, alpha, beta, not is_maximising_player, training, agent, epsilon)
            # Each ply after a checkmate is slower, so they get ranked slightly less
            # We want the fastest mate!
            if curr_move > MATE_THRESHOLD:
                curr_move -= 1
            elif curr_move < -MATE_THRESHOLD:
                curr_move += 1
            best_move = max(
                best_move,
                curr_move,
            )
            board.pop()
            alpha = max(alpha, best_move)
            if beta <= alpha:
                return best_move
        return best_move
    else:
        best_move = float("inf")
        moves = get_ordered_moves(board)
        for move in moves:
            board.push(move)
            curr_move = minimax(depth - 1, board, alpha, beta, not is_maximising_player, training, agent, epsilon)
            if curr_move > MATE_THRESHOLD:
                curr_move -= 1
            elif curr_move < -MATE_THRESHOLD:
                curr_move += 1
            best_move = min(
                best_move,
                curr_move,
            )
            board.pop()
            beta = min(beta, best_move)
            if beta <= alpha:
                return best_move
        return best_move


def act(agent, state, env):
        if numpy.random.rand() <= agent.epsilon:
            #get legal moves
            legalMoves = agent.env.get_board().legal_moves
            legalMoves = list(legalMoves)
            random_move_array, idx = env.encode_move(random.choice(legalMoves), False, agent.env.get_board().turn)
            # print(random_move_array)
            return random_move_array, idx
        
        # filter legal moves
        legalMoves = agent.env.get_board().legal_moves
        legalMoves = list(legalMoves)
        legalMoves = [env.encode_move(move, True, agent.env.get_board().turn)[1] for move in legalMoves]
        actValues = agent.model.predict(state, verbose=1)[0]
        # os.system('cls')
        actValues = [actValues[move] for move in legalMoves]
        mx = legalMoves[numpy.argmax(actValues) if agent.env.get_board().turn else numpy.argmin(actValues)]
        arr = numpy.zeros(shape=[76, 8, 8])
        arr[mx[0]][mx[1]][mx[2]] = 1
        # print(arr)
        return arr, (mx[0], mx[1], mx[2])
    
