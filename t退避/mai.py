from typing import List, Tuple, Optional
from local_driver import Alg3D, Board # ローカル検証用
# from framework import Alg3D, Board # 本番用
import math

class MyAI(Alg3D):
    def __init__(self):
        self.max_depth = 4  # 探索深度（元のまま）
        self.player_num = None  # 自分のプレイヤー番号
        # 勝利パターンを事前計算（ビットマスク）
        self.win_patterns = self._generate_win_patterns()
        
    def get_move(
        self,
        board: List[List[List[int]]], # 盤面情報
        player: int, # 先手(黒):1 後手(白):2
        last_move: Tuple[int, int, int] # 直前に置かれた場所(x, y, z)
    ) -> Tuple[int, int]:
        """
        ビットボード実装の立体４目並べAI（Minimax + Alpha-Beta）
        """
        self.player_num = player
        
        # ビットボードに変換
        black_board, white_board = self._convert_to_bitboard(board)
        
        # 有効な手を取得
        valid_moves = self._get_valid_moves_bb(black_board, white_board)
        if not valid_moves:
            return (0, 0)  # フォールバック
        
        # Minimax + Alpha-Beta探索で最適手を決定（元のアルゴリズム）
        _, best_move = self._alpha_beta_bb(black_board, white_board, self.max_depth, 
                                         -math.inf, math.inf, True, player)
        
        if best_move is None:
            return valid_moves[0]  # フォールバック
        
        return best_move
    
    def _convert_to_bitboard(self, board: List[List[List[int]]]) -> Tuple[int, int]:
        """3次元リストをビットボードに変換"""
        black_board = 0
        white_board = 0
        
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    bit_pos = z * 16 + y * 4 + x
                    if board[z][y][x] == 1:  # 黒
                        black_board |= (1 << bit_pos)
                    elif board[z][y][x] == 2:  # 白
                        white_board |= (1 << bit_pos)
        
        return black_board, white_board
    
    def _get_valid_moves_bb(self, black_board: int, white_board: int) -> List[Tuple[int, int]]:
        """ビットボードから有効な手を取得"""
        valid_moves = []
        occupied = black_board | white_board
        
        for x in range(4):
            for y in range(4):
                # 一番上の層（z=3）が空かチェック
                top_bit_pos = 3 * 16 + y * 4 + x
                if not (occupied & (1 << top_bit_pos)):
                    valid_moves.append((x, y))
        
        return valid_moves
    
    def _alpha_beta_bb(self, black_board: int, white_board: int, depth: int, 
                      alpha: float, beta: float, maximizing_player: bool, 
                      current_player: int) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        ビットボード版Alpha-Betaプルーニング付きのMinimax探索
        """
        # 終了条件をチェック
        if depth == 0 or self._is_terminal_bb(black_board, white_board):
            return self._evaluate_board_bb(black_board, white_board), None
        
        valid_moves = self._get_valid_moves_bb(black_board, white_board)
        if not valid_moves:
            return self._evaluate_board_bb(black_board, white_board), None
        
        best_move = None
        
        if maximizing_player:
            max_eval = -math.inf
            for move in valid_moves:
                x, y = move
                new_black, new_white, z = self._make_move_bb(black_board, white_board, x, y, current_player)
                if z == -1:
                    continue
                
                eval_score, _ = self._alpha_beta_bb(new_black, new_white, depth - 1, alpha, beta, False, 
                                                  2 if current_player == 1 else 1)
                
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
                new_black, new_white, z = self._make_move_bb(black_board, white_board, x, y, current_player)
                if z == -1:
                    continue
                
                eval_score, _ = self._alpha_beta_bb(new_black, new_white, depth - 1, alpha, beta, True, 
                                                  2 if current_player == 1 else 1)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-Betaプルーニング
            
            return min_eval, best_move
    
    def _make_move_bb(self, black_board: int, white_board: int, x: int, y: int, 
                     player: int) -> Tuple[int, int, int]:
        """ビットボードに手を打ち、新しいボードとz座標を返す"""
        occupied = black_board | white_board
        
        # 下から順に空いている位置を探す
        for z in range(4):
            bit_pos = z * 16 + y * 4 + x
            if not (occupied & (1 << bit_pos)):
                if player == 1:  # 黒
                    return black_board | (1 << bit_pos), white_board, z
                else:  # 白
                    return black_board, white_board | (1 << bit_pos), z
        
        return black_board, white_board, -1  # 置けない場合
    
    def _is_terminal_bb(self, black_board: int, white_board: int) -> bool:
        """ビットボード版ゲーム終了判定"""
        # 勝者がいるかチェック
        if self._check_win_bb(black_board) or self._check_win_bb(white_board):
            return True
        
        # 盤面が満杯かチェック
        return len(self._get_valid_moves_bb(black_board, white_board)) == 0
    
    def _check_win_bb(self, board: int) -> bool:
        """ビットボード版勝利判定"""
        # 事前計算した勝利パターンをチェック
        for pattern in self.win_patterns:
            if (board & pattern) == pattern:
                return True
        return False
    
    def _generate_win_patterns(self) -> List[int]:
        """4つ並びの勝利パターンを事前計算"""
        patterns = []
        
        # すべての方向での4つ並びパターンを生成
        directions = [
            (1, 0, 0),   # X軸方向
            (0, 1, 0),   # Y軸方向  
            (0, 0, 1),   # Z軸方向
            (1, 1, 0),   # XY平面の斜め
            (1, -1, 0),
            (1, 0, 1),   # XZ平面の斜め
            (1, 0, -1),
            (0, 1, 1),   # YZ平面の斜め
            (0, 1, -1),
            (1, 1, 1),   # 3D対角線
            (1, 1, -1),
            (1, -1, 1),
            (-1, 1, 1)
        ]
        
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    for dx, dy, dz in directions:
                        pattern = 0
                        valid = True
                        
                        # 4つ並びのパターンを作成
                        for i in range(4):
                            nx, ny, nz = x + i * dx, y + i * dy, z + i * dz
                            if 0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4:
                                bit_pos = nz * 16 + ny * 4 + nx
                                pattern |= (1 << bit_pos)
                            else:
                                valid = False
                                break
                        
                        if valid and pattern not in patterns:
                            patterns.append(pattern)
        
        return patterns
    
    def _evaluate_board_bb(self, black_board: int, white_board: int) -> float:
        """ビットボード版盤面評価関数"""
        if self.player_num is None:
            return 0.0
        
        player_board = black_board if self.player_num == 1 else white_board
        opponent_board = white_board if self.player_num == 1 else black_board
        
        # 勝利状態の特別評価
        if self._check_win_bb(player_board):
            return 1000.0
        if self._check_win_bb(opponent_board):
            return -1000.0
        
        # 連続性の評価
        player_score = self._evaluate_threats_bb(player_board, opponent_board)
        opponent_score = self._evaluate_threats_bb(opponent_board, player_board)
        
        return player_score - opponent_score
    
    def _evaluate_threats_bb(self, my_board: int, opp_board: int) -> float:
        """ビットボード版脅威評価"""
        score = 0.0
        
        # 勝利パターンから脅威を評価
        for pattern in self.win_patterns:
            my_bits = my_board & pattern
            opp_bits = opp_board & pattern
            
            # 相手の石がある場合はスキップ
            if opp_bits:
                continue
            
            # 自分の石の数をカウント
            count = bin(my_bits).count('1')
            
            if count == 3:
                score += 50.0  # 3つ並び
            elif count == 2:
                score += 10.0  # 2つ並び
            elif count == 1:
                score += 1.0   # 1つ
        
        return score

    # 元のアルゴリズムとの互換性を保つための関数群
    def get_valid_moves(self, board: List[List[List[int]]]) -> List[Tuple[int, int]]:
        """互換性のための関数"""
        black_board, white_board = self._convert_to_bitboard(board)
        return self._get_valid_moves_bb(black_board, white_board)
    
    def alpha_beta(self, board: List[List[List[int]]], depth: int, alpha: float, beta: float, 
                   maximizing_player: bool, current_player: int) -> Tuple[float, Optional[Tuple[int, int]]]:
        """互換性のための関数"""
        black_board, white_board = self._convert_to_bitboard(board)
        return self._alpha_beta_bb(black_board, white_board, depth, alpha, beta, maximizing_player, current_player)