"""
Tic Tac Toe Player
"""

import math
import copy


X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    turns = 0

    for list in board:
        for move in list:
            if move != EMPTY:
                turns += 1
    
    return X if turns % 2 == 0 else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()

    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                actions.add((i, j))

    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    #print(action, board)
    if action not in actions(board):
        raise Exception("Move not valid")
    
    board = copy.deepcopy(board)
    board[action[0]][action[1]] = player(board)
    
    return board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for i in range(3):
        if board[0][i] == board[1][i] == board[2][i]:
            if board[0][i] == X:
                return X
            if board[0][i] == O:
                return O    
        if board[i][0] == board[i][1] == board[i][2]:
            if board[i][0] == X:
                return X
            if board[i][0] == O:
                return O

    if board[0][0] == board[1][1] == board[2][2]:
        if board[0][0] == X:
            return X
        if board[0][0] == O:
            return O
    
    if board[0][2] == board[1][1] == board[2][0]:
        if board[0][2] == X:
                return X
        if board[0][2] == O:
            return O
    
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    
    for row in board:
        if EMPTY in row:
            return False
    
    return True

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board) is True:
        return None

    if player(board) is X:
        return max_score(board, float("inf"))[1]
    else:
        return min_score(board, float("-inf"))[1]


def max_score(board, min):

    if terminal(board) is True:
        return [utility(board), None]

    max = float("-inf")
    move = None
    for action in actions(board):
        value = min_score(result(board, action), max)[0]
        if value > max:
            max = value
            move = action
        if value > min:
           break
    
    return [max, move]


def min_score(board, max):

    if terminal(board) is True:
        return [utility(board), None]

    min = float("inf")
    move = None
    for action in actions(board):
        value = max_score(result(board, action), min)[0]
        if value < min:
            min = value
            move = action
        if value < max:
           break
    
    return [min, move]