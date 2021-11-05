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

    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == EMPTY:
                actions.add((i, j))

    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    print(board, action)
    if action not in actions(board):
        raise Exception("Move not valid")
    
    new_board = copy.deepcopy(board)
    new_board[action[0]][action[1]] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
   
    if [X, X, X] in board:
        return X
    if [O, O, O] in board:
        return O

    for i in range(len(board)):
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

    moves = actions(board)
    if len(moves) == 1:
        return moves[0]

        
    scores = {}

    def get_scores(scores, board):
        for action in actions(board):

            current_player = player(board)
            current_board = result(board, action)

            # update score            
            if len(moves) == 1:
                return utility(current_board)
            else:
                scores[action] += get_scores(scores, current_board)

            return scores

            # choose best score option

    get_scores(moves, scores, board)
