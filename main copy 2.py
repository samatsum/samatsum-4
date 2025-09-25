from typing import List, Tuple
# from local_driver import Alg3D, Board # ローカル検証用
from framework import Alg3D, Board # 本番用

class MyAI(Alg3D):
    def get_move(
        self,
        board: List[List[List[int]]], # 盤面情報
        player: int, # 先手(黒):1 後手(白):2
        last_move: Tuple[int, int, int] # 直前に置かれた場所(x, y, z)
    ) -> Tuple[int, int]:
        """
        立体４目並べのAIアルゴリズム
        優先順位：
        1. 勝利手がある場合はそれを選ぶ
        2. 相手の勝利を阻止する
        3. センター付近を優先
        4. 有効な手をランダムに選ぶ
        """
        
        # 有効な手を取得
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return (0, 0)  # フォールバック
        
        # 1. 勝利手をチェック
        winning_move = self.find_winning_move(board, player, valid_moves)
        if winning_move:
            return winning_move
        
        # 2. 相手の勝利を阻止
        opponent = 2 if player == 1 else 1
        blocking_move = self.find_winning_move(board, opponent, valid_moves)
        if blocking_move:
            return blocking_move
        
        # 3. センター付近を優先
        center_move = self.find_center_move(valid_moves)
        if center_move:
            return center_move
        
        # 4. 最初の有効な手を返す
        return valid_moves[0]
    
    def get_valid_moves(self, board: List[List[List[int]]]) -> List[Tuple[int, int]]:
        """有効な手（石を置ける場所）を取得"""
        valid_moves = []
        for x in range(4):
            for y in range(4):
                # 列が満杯でないかチェック
                if board[3][y][x] == 0:  # 一番上が空なら置ける
                    valid_moves.append((x, y))
        return valid_moves
    
    def simulate_move(self, board: List[List[List[int]]], x: int, y: int, player: int) -> int:
        """指定した位置に石を置いた場合のz座標を返す"""
        for z in range(4):
            if board[z][y][x] == 0:
                return z
        return -1  # 置けない場合
    
    def find_winning_move(self, board: List[List[List[int]]], player: int, valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """勝利につながる手を見つける"""
        for x, y in valid_moves:
            # 仮想的に石を置いてみる
            z = self.simulate_move(board, x, y, player)
            if z == -1:
                continue
            
            # 一時的に石を置く
            board[z][y][x] = player
            
            # 勝利条件をチェック
            if self.check_win(board, x, y, z, player):
                board[z][y][x] = 0  # 元に戻す
                return (x, y)
            
            board[z][y][x] = 0  # 元に戻す
        
        return None
    
    def check_win(self, board: List[List[List[int]]], x: int, y: int, z: int, player: int) -> bool:
        """指定した位置から4つ並んでいるかチェック"""
        directions = [
            # X軸方向
            (1, 0, 0), (-1, 0, 0),
            # Y軸方向  
            (0, 1, 0), (0, -1, 0),
            # Z軸方向
            (0, 0, 1), (0, 0, -1),
            # XY平面の斜め
            (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
            # XZ平面の斜め
            (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
            # YZ平面の斜め
            (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1),
            # 3D対角線
            (1, 1, 1), (-1, -1, -1), (1, 1, -1), (-1, -1, 1),
            (1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1)
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
    
    def find_center_move(self, valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """センター付近の手を優先して返す"""
        # センターからの距離でソート
        center_x, center_y = 1.5, 1.5
        
        def distance_from_center(move):
            x, y = move
            return abs(x - center_x) + abs(y - center_y)
        
        sorted_moves = sorted(valid_moves, key=distance_from_center)
        return sorted_moves[0] if sorted_moves else None