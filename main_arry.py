from typing import List, Tuple, Optional
from local_driver import Alg3D, Board # ローカル検証用
# from framework import Alg3D, Board # 本番用
import math

class MyAI(Alg3D):
    def __init__(self):
        self.max_depth = 4  # 探索深度
        self.player_num = None  # 自分のプレイヤー番号
        
    def get_move(
        self,
        board: List[List[List[int]]], # 盤面情報
        player: int, # 先手(黒):1 後手(白):2
        last_move: Tuple[int, int, int] # 直前に置かれた場所(x, y, z)
    ) -> Tuple[int, int]:
        """
        探索ベースの立体４目並べAI（Minimax + Alpha-Beta）
        """
        self.player_num = player
        
        # 有効な手を取得
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return (0, 0)  # フォールバック
        
        # Minimax + Alpha-Beta探索で最適手を決定
        _, best_move = self.alpha_beta(board, self.max_depth, -math.inf, math.inf, True, player)
        
        if best_move is None:
            return valid_moves[0]  # フォールバック
        
        return best_move
    
    def alpha_beta(self, board: List[List[List[int]]], depth: int, alpha: float, beta: float, 
                   maximizing_player: bool, current_player: int) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Alpha-Betaプルーニング付きのMinimax探索
        """
        # 終了条件をチェック
        if depth == 0 or self.is_terminal(board):
            return self.evaluate_board(board), None
        
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return self.evaluate_board(board), None
        
        best_move = None
        
        if maximizing_player:
            max_eval = -math.inf
            for move in valid_moves:
                x, y = move
                z = self.make_move(board, x, y, current_player)
                if z == -1:
                    continue
                
                eval_score, _ = self.alpha_beta(board, depth - 1, alpha, beta, False, 
                                              2 if current_player == 1 else 1)
                self.undo_move(board, x, y, z)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-Betaプルーニング
            
            return max_eval, best_move
        else:
            min_eval = math.inf
            for move in valid_moves:
                x, y = move
                z = self.make_move(board, x, y, current_player)
                if z == -1:
                    continue
                
                eval_score, _ = self.alpha_beta(board, depth - 1, alpha, beta, True, 
                                              2 if current_player == 1 else 1)
                self.undo_move(board, x, y, z)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-Betaプルーニング
            
            return min_eval, best_move
    
    def get_valid_moves(self, board: List[List[List[int]]]) -> List[Tuple[int, int]]:
        """有効な手（石を置ける場所）を取得"""
        valid_moves = []
        for x in range(4):
            for y in range(4):
                # 列が満杯でないかチェック
                if board[3][y][x] == 0:  # 一番上が空なら置ける
                    valid_moves.append((x, y))
        return valid_moves
    
    def make_move(self, board: List[List[List[int]]], x: int, y: int, player: int) -> int:
        """指定した位置に石を置き、実際のz座標を返す"""
        for z in range(4):
            if board[z][y][x] == 0:
                board[z][y][x] = player
                return z
        return -1  # 置けない場合
    
    def undo_move(self, board: List[List[List[int]]], x: int, y: int, z: int) -> None:
        """手を取り消す"""
        board[z][y][x] = 0
    
    def is_terminal(self, board: List[List[List[int]]]) -> bool:
        """ゲーム終了判定"""
        # 勝者がいるかチェック
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    if board[z][y][x] != 0:
                        if self.check_win_from_position(board, x, y, z, board[z][y][x]):
                            return True
        
        # 盤面が満杯かチェック
        return len(self.get_valid_moves(board)) == 0
    
    def evaluate_board(self, board: List[List[List[int]]]) -> float:
        """盤面評価関数"""
        if self.player_num is None:
            return 0.0
        
        player_score = 0.0
        opponent_score = 0.0
        opponent = 2 if self.player_num == 1 else 1
        
        # 各方向での連続数をカウントして評価
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    if board[z][y][x] != 0:
                        player_val = board[z][y][x]
                        if player_val == self.player_num:
                            player_score += self.evaluate_position(board, x, y, z, player_val)
                        else:
                            opponent_score += self.evaluate_position(board, x, y, z, player_val)
        
        # 勝利状態の特別評価
        if self.check_win_any(board, self.player_num):
            return 1000.0
        if self.check_win_any(board, opponent):
            return -1000.0
        
        return player_score - opponent_score
    
    def evaluate_position(self, board: List[List[List[int]]], x: int, y: int, z: int, player: int) -> float:
        """指定位置からの評価値を計算"""
        directions = [
            # X軸方向
            (1, 0, 0), # Y軸方向  
            (0, 1, 0), # Z軸方向
            (0, 0, 1), # XY平面の斜め
            (1, 1, 0), (1, -1, 0), # XZ平面の斜め
            (1, 0, 1), (1, 0, -1), # YZ平面の斜め
            (0, 1, 1), (0, 1, -1), # 3D対角線
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)
        ]
        
        total_score = 0.0
        
        for dx, dy, dz in directions:
            line_score = self.evaluate_line(board, x, y, z, dx, dy, dz, player)
            total_score += line_score
        
        return total_score
    
    def evaluate_line(self, board: List[List[List[int]]], x: int, y: int, z: int, 
                     dx: int, dy: int, dz: int, player: int) -> float:
        """一方向での連続性を評価"""
        count = 1  # 現在の石を含む
        blocked = 0  # 両端がブロックされているかどうか
        
        # 正方向
        nx, ny, nz = x + dx, y + dy, z + dz
        while (0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4 and 
               board[nz][ny][nx] == player):
            count += 1
            nx, ny, nz = nx + dx, ny + dy, nz + dz
        
        if not (0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4) or board[nz][ny][nx] != 0:
            blocked += 1
        
        # 負方向
        nx, ny, nz = x - dx, y - dy, z - dz
        while (0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4 and 
               board[nz][ny][nx] == player):
            count += 1
            nx, ny, nz = nx - dx, ny - dy, nz - dz
        
        if not (0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4) or board[nz][ny][nx] != 0:
            blocked += 1
        
        # スコア計算
        if count >= 4:
            return 1000.0  # 勝利
        elif count == 3 and blocked < 2:
            return 50.0
        elif count == 2 and blocked < 2:
            return 10.0
        elif count == 1 and blocked < 2:
            return 1.0
        
        return 0.0
    
    def check_win_from_position(self, board: List[List[List[int]]], x: int, y: int, z: int, player: int) -> bool:
        """指定した位置から4つ並んでいるかチェック"""
        directions = [
            # X軸方向
            (1, 0, 0), # Y軸方向  
            (0, 1, 0), # Z軸方向
            (0, 0, 1), # XY平面の斜め
            (1, 1, 0), (1, -1, 0), # XZ平面の斜め
            (1, 0, 1), (1, 0, -1), # YZ平面の斜め
            (0, 1, 1), (0, 1, -1), # 3D対角線
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)
        ]
        
        for dx, dy, dz in directions:
            count = 1  # 現在の石を含む
            
            # 正方向にカウント
            nx, ny, nz = x + dx, y + dy, z + dz
            while (0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4 and 
                   board[nz][ny][nx] == player):
                count += 1
                nx, ny, nz = nx + dx, ny + dy, nz + dz
            
            # 負方向にカウント
            nx, ny, nz = x - dx, y - dy, z - dz
            while (0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4 and 
                   board[nz][ny][nx] == player):
                count += 1
                nx, ny, nz = nx - dx, ny - dy, nz - dz
            
            if count >= 4:
                return True
        
        return False
    
    def check_win_any(self, board: List[List[List[int]]], player: int) -> bool:
        """盤面全体で指定プレイヤーが勝利しているかチェック"""
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    if board[z][y][x] == player:
                        if self.check_win_from_position(board, x, y, z, player):
                            return True
        return False

#1:54
#  		1手目  お試し : 黒 : (1, 1)

# 2手目  お試し : 白 : (1, 1)

# 3手目  お試し : 黒 : (1, 2)

# 4手目  お試し : 白 : (0, 0)  ※異常終了したため、 (0, 0)に強制配置

# 5手目  お試し : 黒 : (2, 1)

# 6手目  お試し : 白 : (0, 0)

# 7手目  お試し : 黒 : (1, 2)

# 8手目  お試し : 白 : (0, 0)

# 9手目  お試し : 黒 : (0, 0)

# 10手目  お試し : 白 : (1, 2)

# 11手目  お試し : 黒 : (1, 0)

# 12手目  お試し : 白 : (1, 3)

# 13手目  お試し : 黒 : (0, 1)

# 14手目  お試し : 白 : (1, 0)  ※異常終了したため、 (1, 0)に強制配置

# 15手目  お試し : 黒 : (1, 0)  ※異常終了したため、 (1, 0)に強制配置

# 16手目  お試し : 白 : (1, 0)  ※異常終了したため、 (1, 0)に強制配置

# 17手目  お試し : 黒 : (2, 0)  ※異常終了したため、 (2, 0)に強制配置

# 18手目  お試し : 白 : (2, 0)  ※異常終了したため、 (2, 0)に強制配置

# 19手目  お試し : 黒 : (0, 3)

# 20手目  お試し : 白 : (2, 0)  ※異常終了したため、 (2, 0)に強制配置

# 21手目  お試し : 黒 : (0, 1)

# 22手目  お試し : 白 : (0, 1)

# 23手目  お試し : 黒 : (0, 1)

# 24手目  お試し : 白 : (2, 0)  ※異常終了したため、 (2, 0)に強制配置

# 25手目  お試し : 黒 : (0, 2)

# 26手目  お試し : 白 : (3, 0)  ※異常終了したため、 (3, 0)に強制配置

# 27手目  お試し : 黒 : (3, 0)  ※異常終了したため、 (3, 0)に強制配置

# 28手目  お試し : 白 : (3, 0)  ※異常終了したため、 (3, 0)に強制配置

# 29手目  お試し : 黒 : (3, 0)  ※異常終了したため、 (3, 0)に強制配置

# 30手目  お試し : 白 : (0, 2)

# 31手目  お試し : 黒 : (0, 3)

# 32手目  お試し : 白 : (0, 2)

# 33手目  お試し : 黒 : (0, 3)

# 34手目  お試し : 白 : (0, 2)

# 35手目  お試し : 黒 : (0, 3)

# 🎉 お試し が 35手で勝利！

