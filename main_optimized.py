from typing import List, Tuple, Optional
from local_driver import Alg3D, Board # ローカル検証用
# from framework import Alg3D, Board # 本番用
import math

class MyAI(Alg3D):
    def __init__(self):
        self.max_depth = 5  # 深度を1つ上げる
        self.player_num = None
        
        # 勝利パターンをタプルで事前計算（リストより高速）
        self.win_patterns = tuple(self._generate_win_patterns_optimized())
        
        # 位置評価テーブル（中央ほど高評価）
        self.position_values = self._create_position_table()
        
        # ビット操作用の定数
        self.ALL_BITS = (1 << 64) - 1
        
        # 置換表（トランスポジションテーブル）
        self.transposition_table = {}
        self.max_table_size = 100000  # メモリ制限内で調整
        
    def get_move(
        self,
        board: List[List[List[int]]], 
        player: int, 
        last_move: Tuple[int, int, int]
    ) -> Tuple[int, int]:
        """
        超最適化版立体４目並べAI
        """
        self.player_num = player
        
        # ビットボードに変換（最適化版）
        black_board, white_board = self._convert_to_bitboard_fast(board)
        
        # 序盤定石チェック
        move_count = bin(black_board | white_board).count('1')
        if move_count < 4:
            opening_move = self._get_opening_move(black_board, white_board, player)
            if opening_move:
                return opening_move
        
        # 必勝・必敗手の即座判定
        immediate_win = self._find_immediate_win(black_board, white_board, player)
        if immediate_win:
            return immediate_win
            
        immediate_block = self._find_immediate_block(black_board, white_board, player)
        if immediate_block:
            return immediate_block
        
        # 有効手生成（move ordering付き）
        valid_moves = self._get_ordered_moves(black_board, white_board)
        if not valid_moves:
            return (0, 0)
        
        # 反復深化 + Alpha-Beta + 置換表
        best_move = None
        for depth in range(1, self.max_depth + 1):
            try:
                _, move = self._alpha_beta_optimized(
                    black_board, white_board, depth, 
                    -math.inf, math.inf, True, player, 0
                )
                if move:
                    best_move = move
            except:  # 時間制限対策
                break
        
        return best_move if best_move else valid_moves[0]
    
    def _convert_to_bitboard_fast(self, board: List[List[List[int]]]) -> Tuple[int, int]:
        """高速ビットボード変換（ループ最適化）"""
        black_board = 0
        white_board = 0
        
        # ネストループを単一ループに変換
        for pos in range(64):
            z, remainder = divmod(pos, 16)
            y, x = divmod(remainder, 4)
            
            cell = board[z][y][x]
            if cell == 1:
                black_board |= (1 << pos)
            elif cell == 2:
                white_board |= (1 << pos)
        
        return black_board, white_board
    
    def _get_opening_move(self, black_board: int, white_board: int, player: int) -> Optional[Tuple[int, int]]:
        """序盤定石"""
        occupied = black_board | white_board
        
        # 初手は中央付近
        if not occupied:
            return (1, 1)  # または (2, 2)
        
        # 2手目以降は相手の近くか中央
        center_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        for x, y in center_positions:
            if self._is_valid_move_fast(occupied, x, y):
                return (x, y)
        
        return None
    
    def _find_immediate_win(self, black_board: int, white_board: int, player: int) -> Optional[Tuple[int, int]]:
        """即座に勝てる手を探す"""
        my_board = black_board if player == 1 else white_board
        opp_board = white_board if player == 1 else black_board
        occupied = black_board | white_board
        
        for x in range(4):
            for y in range(4):
                if not self._is_valid_move_fast(occupied, x, y):
                    continue
                
                # 仮に置いてみる
                z = self._get_drop_level_fast(occupied, x, y)
                if z == -1:
                    continue
                    
                bit_pos = z * 16 + y * 4 + x
                test_board = my_board | (1 << bit_pos)
                
                if self._check_win_fast(test_board):
                    return (x, y)
        
        return None
    
    def _find_immediate_block(self, black_board: int, white_board: int, player: int) -> Optional[Tuple[int, int]]:
        """相手の即座勝利を阻止する手"""
        opp_player = 2 if player == 1 else 1
        return self._find_immediate_win(black_board, white_board, opp_player)
    
    def _get_ordered_moves(self, black_board: int, white_board: int) -> List[Tuple[int, int]]:
        """Move Ordering付きの有効手生成"""
        occupied = black_board | white_board
        moves_with_score = []
        
        for x in range(4):
            for y in range(4):
                if not self._is_valid_move_fast(occupied, x, y):
                    continue
                
                # 位置評価を付けて並び替え
                score = self.position_values[y][x]
                
                # 中央により近い手を優先
                center_distance = abs(x - 1.5) + abs(y - 1.5)
                score -= center_distance
                
                moves_with_score.append((score, (x, y)))
        
        # スコア順でソート（高い順）
        moves_with_score.sort(reverse=True)
        return [move for _, move in moves_with_score]
    
    def _alpha_beta_optimized(self, black_board: int, white_board: int, depth: int, 
                             alpha: float, beta: float, maximizing_player: bool, 
                             current_player: int, ply: int) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        超最適化版Alpha-Beta（置換表 + Move Ordering + 枝刈り強化）
        """
        # 置換表チェック
        board_key = (black_board, white_board, depth, maximizing_player)
        if board_key in self.transposition_table:
            cached_eval, cached_move = self.transposition_table[board_key]
            return cached_eval, cached_move
        
        # 終了条件
        if depth == 0 or self._is_terminal_fast(black_board, white_board):
            eval_score = self._evaluate_board_optimized(black_board, white_board, ply)
            return eval_score, None
        
        # 有効手を取得（すでにソート済み）
        valid_moves = self._get_ordered_moves(black_board, white_board)
        if not valid_moves:
            eval_score = self._evaluate_board_optimized(black_board, white_board, ply)
            return eval_score, None
        
        best_move = None
        
        if maximizing_player:
            max_eval = -math.inf
            for move in valid_moves:
                x, y = move
                new_black, new_white, z = self._make_move_fast(black_board, white_board, x, y, current_player)
                if z == -1:
                    continue
                
                eval_score, _ = self._alpha_beta_optimized(
                    new_black, new_white, depth - 1, alpha, beta, False, 
                    2 if current_player == 1 else 1, ply + 1
                )
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-Beta cut-off
            
            result = (max_eval, best_move)
        else:
            min_eval = math.inf
            for move in valid_moves:
                x, y = move
                new_black, new_white, z = self._make_move_fast(black_board, white_board, x, y, current_player)
                if z == -1:
                    continue
                
                eval_score, _ = self._alpha_beta_optimized(
                    new_black, new_white, depth - 1, alpha, beta, True, 
                    2 if current_player == 1 else 1, ply + 1
                )
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-Beta cut-off
            
            result = (min_eval, best_move)
        
        # 置換表に保存（サイズ制限）
        if len(self.transposition_table) < self.max_table_size:
            self.transposition_table[board_key] = result
        
        return result
    
    def _is_valid_move_fast(self, occupied: int, x: int, y: int) -> bool:
        """高速有効手判定"""
        top_bit_pos = 3 * 16 + y * 4 + x
        return not (occupied & (1 << top_bit_pos))
    
    def _get_drop_level_fast(self, occupied: int, x: int, y: int) -> int:
        """高速落下レベル取得"""
        for z in range(4):
            bit_pos = z * 16 + y * 4 + x
            if not (occupied & (1 << bit_pos)):
                return z
        return -1
    
    def _make_move_fast(self, black_board: int, white_board: int, x: int, y: int, 
                       player: int) -> Tuple[int, int, int]:
        """高速手生成"""
        occupied = black_board | white_board
        z = self._get_drop_level_fast(occupied, x, y)
        
        if z == -1:
            return black_board, white_board, -1
        
        bit_pos = z * 16 + y * 4 + x
        if player == 1:
            return black_board | (1 << bit_pos), white_board, z
        else:
            return black_board, white_board | (1 << bit_pos), z
    
    def _is_terminal_fast(self, black_board: int, white_board: int) -> bool:
        """高速終了判定"""
        return (self._check_win_fast(black_board) or 
                self._check_win_fast(white_board) or 
                (black_board | white_board) == self.ALL_BITS)
    
    def _check_win_fast(self, board: int) -> bool:
        """高速勝利判定（ビット演算最適化）"""
        # 事前計算したパターンと高速ビット演算
        for pattern in self.win_patterns:
            if (board & pattern) == pattern:
                return True
        return False
    
    def _generate_win_patterns_optimized(self) -> List[int]:
        """最適化された勝利パターン生成"""
        patterns = set()  # 重複除去
        
        # より効率的な方向ベクトル
        directions = (
            (1, 0, 0), (0, 1, 0), (0, 0, 1),    # 軸方向
            (1, 1, 0), (1, -1, 0),               # XY斜め
            (1, 0, 1), (1, 0, -1),               # XZ斜め
            (0, 1, 1), (0, 1, -1),               # YZ斜め
            (1, 1, 1), (1, 1, -1),               # 3D対角線
            (1, -1, 1), (-1, 1, 1)
        )
        
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    for dx, dy, dz in directions:
                        pattern = 0
                        valid = True
                        
                        for i in range(4):
                            nx, ny, nz = x + i * dx, y + i * dy, z + i * dz
                            if 0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4:
                                bit_pos = nz * 16 + ny * 4 + nx
                                pattern |= (1 << bit_pos)
                            else:
                                valid = False
                                break
                        
                        if valid:
                            patterns.add(pattern)
        
        return list(patterns)
    
    def _create_position_table(self) -> List[List[float]]:
        """位置評価テーブル作成"""
        table = []
        for y in range(4):
            row = []
            for x in range(4):
                # 中央ほど高い評価
                center_distance = abs(x - 1.5) + abs(y - 1.5)
                value = 10.0 - center_distance
                row.append(value)
            table.append(row)
        return table
    
    def _evaluate_board_optimized(self, black_board: int, white_board: int, ply: int) -> float:
        """最適化された評価関数"""
        if self.player_num is None:
            return 0.0
        
        player_board = black_board if self.player_num == 1 else white_board
        opponent_board = white_board if self.player_num == 1 else black_board
        
        # 勝利状態の特別評価（手数考慮）
        if self._check_win_fast(player_board):
            return 10000.0 - ply  # 早い勝利ほど高評価
        if self._check_win_fast(opponent_board):
            return -10000.0 + ply  # 遅い敗北ほど高評価
        
        # 高速脅威評価
        player_score = self._evaluate_threats_optimized(player_board, opponent_board)
        opponent_score = self._evaluate_threats_optimized(opponent_board, player_board)
        
        # 位置評価を追加
        position_score = self._evaluate_positions(player_board, opponent_board)
        
        return player_score - opponent_score + position_score
    
    def _evaluate_threats_optimized(self, my_board: int, opp_board: int) -> float:
        """最適化された脅威評価（ルックアップテーブル使用）"""
        score = 0.0
        
        # パターンマッチングの最適化
        for pattern in self.win_patterns:
            my_bits = my_board & pattern
            if opp_board & pattern:  # 相手の石があるパターンはスキップ
                continue
            
            # ポップカウント（ビット数計算）の最適化
            count = my_bits.bit_count() if hasattr(int, 'bit_count') else bin(my_bits).count('1')
            
            # スコアテーブル
            if count == 3:
                score += 100.0
            elif count == 2:
                score += 15.0
            elif count == 1:
                score += 2.0
        
        return score
    
    def _evaluate_positions(self, player_board: int, opponent_board: int) -> float:
        """位置評価"""
        score = 0.0
        
        for pos in range(64):
            z, remainder = divmod(pos, 16)
            y, x = divmod(remainder, 4)
            
            bit_mask = 1 << pos
            if player_board & bit_mask:
                score += self.position_values[y][x]
            elif opponent_board & bit_mask:
                score -= self.position_values[y][x]
        
        return score * 0.1  # 重み調整

    # 互換性維持のための関数群
    def get_valid_moves(self, board: List[List[List[int]]]) -> List[Tuple[int, int]]:
        black_board, white_board = self._convert_to_bitboard_fast(board)
        return self._get_ordered_moves(black_board, white_board)
