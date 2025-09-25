from typing import List, Tuple
#from local_driver import Alg3D, Board # ローカル検証用
from framework import Alg3D, Board # 本番用
import math

class MyAI(Alg3D):
    def get_move(
        self,
        board: List[List[List[int]]], # 盤面情報
        player: int, # 先手(黒):1 後手(白):2
        last_move: Tuple[int, int, int] # 直前に置かれた場所(x, y, z)
    ) -> Tuple[int, int]:
        # ここにアルゴリズムを書く
        print(self.is_terminal(board, player))
        return (0, 0)

    BOARD_SIZE_X, BOARD_SIZE_Y = 4, 4
    BOARD_SIZE_Z = 4

#     fonction MINIMAX( p) est

    # TO DO
        # def evaluation(self):
            # (+100 pour la victoire de MAX, 0 pour une partie nulle, et -100 pour la victoire de MIN

        # def result(self):
    def generate_lines(self):
        lines = []
        rng = range(4)
        for z in rng:
            for y in rng:
                lines.append([(x,y,z) for x in rng])
        for z in rng:
            for x in rng:
                lines.append([(x,y,z) for y in rng])
        for y in rng:
            for x in rng:
                lines.append([(x,y,z) for z in rng])



        for z in rng:
            lines.append([(i,i,z) for i in rng])
            lines.append([(i,3-i,z) for i in rng])
        for y in rng:
            lines.append([(i,y,i) for i in rng])
            lines.append([(i,y,3-i) for i in rng])
        for x in rng:
            lines.append([(x,i,i) for i in rng])
            lines.append([(x,i,3-i) for i in rng])

        # diagonal
        lines.append([(i,i,i) for i in rng])
        lines.append([(i,i,3-i) for i in rng])
        lines.append([(i,3-i,i) for i in rng])
        lines.append([(3-i,i,i) for i in rng])
        print("NUMBER LINES", lines)
        return lines
    def is_terminal(self, board, player):
        """
            check if the game ended
            return 1 if ai win -1 if lose and 0 equal
        """
        enemy = 1 if player == 2 else 2

        lines = self.generate_lines()
        for line in lines:
            if all(board[z][y][x] == player for (x,y,z) in line):
                return 1
            elif all(board[z][y][x] == enemy for (x,y,z) in line):
                return -1
        return 0

    def legal_move(self, board):
        """
            use to determine how many legal moves are possible
        """
        action_arr = []

        for plane_i in range(board.length):
            for row_i in range(plane_i):
                for space_i in range(row_i):
                    if board[plane_i][row_i][space_i] > 0 \
                        and (board.length - 1 == plane_i \
                        or board[plane_i + 1][row_i][space_i] == 0 ):

                        action_arr.append(tuple(plane_i, row_i, space_i))
        return action_arr

    def minimax(self, board, isMaximiser, depth, max_depth, player):
        """
            isMaximiser: is the computer turn to check in the three
            depth: how far in the three you are
            max_deth: maximmum depth
        """
        if depth == 0 or self.is_terminal(board) or depth == max_depth:
            return self.evaluate(board)
    
        if isMaximiser:
            max_eval = -math.inf
            for action in self.legal_move(board):
                eval = self.minimax(self.result(board, action), False, depth - 1, max_depth)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = math.inf
            for action in self.legal_move(board):
                eval = self.minimax(self.result(board, action), True, depth - 1, max_depth)
                min_eval = min(min_eval, eval)
            return min_eval