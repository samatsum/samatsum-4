from typing import List, Tuple, Optional, Dict
# from local_driver import Alg3D, Board # ローカル検証用
from framework import Alg3D, Board # 本番用
import math
import random
from dataclasses import dataclass

@dataclass
class TranspositionEntry:
    """
    置換表（Transposition Table）のエントリを表すクラス
    - hash_key: ゾブリストハッシュ値（ハッシュ衝突検出用）
    - depth: この評価値を計算した際の探索深度
    - score: 盤面の評価値
    - flag: 評価値のタイプ（EXACT, LOWERBOUND, UPPERBOUND）
    - best_move: この盤面での最善手
    """
    hash_key: int
    depth: int 
    score: float
    flag: str  # "EXACT", "LOWERBOUND", "UPPERBOUND"
    best_move: Optional[Tuple[int, int]]

class MyAI(Alg3D):
    def __init__(self):
        self.max_depth = 4  # 探索深度
        self.player_num = None  # 自分のプレイヤー番号
        
        # 勝利パターンを事前計算（ビットマスク）
        self.win_patterns = self._generate_win_patterns()
        
        # ===== ゾブリストハッシュの初期化 =====
        # 4x4x4 = 64位置 × 2プレイヤー = 128個のランダム値を生成
        self.zobrist_table = self._initialize_zobrist_table()
        
        # ===== 置換表（Transposition Table）の初期化 =====
        # 辞書形式で実装。実用では固定サイズのハッシュテーブルを使うことも多い
        self.transposition_table: Dict[int, TranspositionEntry] = {}
        
        # 置換表のヒット数統計（デバッグ用）
        self.tt_hits = 0
        self.tt_queries = 0
        
    def _initialize_zobrist_table(self) -> List[List[int]]:
        """
        ゾブリストハッシュテーブルを初期化
        
        ゾブリストハッシュとは：
        - 各盤面位置とプレイヤーの組み合わせに対して、ランダムな64ビット数を割り当て
        - 盤面のハッシュ値は、そこに置かれている全ての石に対応する値のXORで計算
        - 手を打つ/戻すときは、対応する値をXORするだけで高速にハッシュ値を更新可能
        
        戻り値：zobrist_table[position][player] = random_value の2次元リスト
        """
        random.seed(42)  # 再現性のため固定シード使用
        zobrist_table = []
        
        # 64位置分（4x4x4）のランダム値を生成
        for position in range(64):
            player_values = []
            for player in range(2):  # プレイヤー0（黒）、プレイヤー1（白）
                # 64ビットのランダム値を生成
                random_value = random.randint(0, (1 << 63) - 1)
                player_values.append(random_value)
            zobrist_table.append(player_values)
        
        return zobrist_table
    
    def _compute_zobrist_hash(self, black_board: int, white_board: int) -> int:
        """
        ビットボードからゾブリストハッシュ値を計算
        
        処理の流れ：
        1. 黒石が置かれている位置を特定
        2. 各位置に対応するゾブリスト値をXOR
        3. 白石についても同様に処理
        4. 全てのXORの結果がハッシュ値
        """
        hash_value = 0
        
        # 黒石（プレイヤー1）のハッシュ値を計算
        temp_black = black_board
        while temp_black:
            # 最下位の1ビットの位置を取得
            position = (temp_black & -temp_black).bit_length() - 1
            # 対応するゾブリスト値をXOR
            hash_value ^= self.zobrist_table[position][0]  # プレイヤー0（黒）
            # 処理済みのビットをクリア
            temp_black &= temp_black - 1
        
        # 白石（プレイヤー2）のハッシュ値を計算
        temp_white = white_board
        while temp_white:
            position = (temp_white & -temp_white).bit_length() - 1
            hash_value ^= self.zobrist_table[position][1]  # プレイヤー1（白）
            temp_white &= temp_white - 1
            
        return hash_value
    
    def _lookup_transposition_table(self, hash_key: int, depth: int, alpha: float, beta: float) -> Tuple[bool, float, Optional[Tuple[int, int]]]:
        """
        置換表から過去の探索結果を検索
        
        置換表の基本概念：
        - 以前に評価した盤面の結果を保存しておく
        - 同じ盤面に再度遭遇した際、再計算せずに保存された結果を利用
        - 探索の重複を削減し、大幅な高速化を実現
        
        引数：
            hash_key: 盤面のゾブリストハッシュ値
            depth: 現在の探索深度
            alpha, beta: Alpha-Beta探索のウィンドウ
        
        戻り値：
            (found, score, best_move) のタプル
            - found: 有効な結果が見つかったかどうか
            - score: 評価値（見つからない場合は0）
            - best_move: 最善手（見つからない場合はNone）
        """
        self.tt_queries += 1
        
        if hash_key not in self.transposition_table:
            return False, 0.0, None
        
        entry = self.transposition_table[hash_key]
        
        # ハッシュ衝突の検証（念のため）
        if entry.hash_key != hash_key:
            return False, 0.0, None
        
        # 保存された探索深度が現在の探索深度以上の場合のみ使用
        if entry.depth < depth:
            return False, 0.0, None
        
        # 評価値のタイプに応じて利用可能性を判定
        if entry.flag == "EXACT":
            # 正確な値の場合はそのまま使用可能
            self.tt_hits += 1
            return True, entry.score, entry.best_move
        elif entry.flag == "LOWERBOUND" and entry.score >= beta:
            # 下限値がbeta以上の場合、beta cutoff可能
            self.tt_hits += 1
            return True, entry.score, entry.best_move
        elif entry.flag == "UPPERBOUND" and entry.score <= alpha:
            # 上限値がalpha以下の場合、alpha cutoff可能
            self.tt_hits += 1
            return True, entry.score, entry.best_move
        
        # 使用できない場合でも、best_moveの情報は有用
        return False, 0.0, entry.best_move
    
    def _store_transposition_table(self, hash_key: int, depth: int, score: float, 
                                  alpha: float, beta: float, best_move: Optional[Tuple[int, int]]):
        """
        置換表に探索結果を保存
        
        評価値のタイプ分類：
        - EXACT: 正確な値（alpha < score < beta）
        - LOWERBOUND: 下限値（score >= beta、beta cutoffが発生）
        - UPPERBOUND: 上限値（score <= alpha、実際の値はこれ以下）
        """
        # 評価値のタイプを決定
        if score <= alpha:
            flag = "UPPERBOUND"
        elif score >= beta:
            flag = "LOWERBOUND"  
        else:
            flag = "EXACT"
        
        # エントリを作成・保存
        entry = TranspositionEntry(
            hash_key=hash_key,
            depth=depth,
            score=score,
            flag=flag,
            best_move=best_move
        )
        
        self.transposition_table[hash_key] = entry

    def get_move(
        self,
        board: List[List[List[int]]], # 盤面情報
        player: int, # 先手(黒):1 後手(白):2
        last_move: Tuple[int, int, int] # 直前に置かれた場所(x, y, z)
    ) -> Tuple[int, int]:
        """
        置換表+ゾブリストハッシュ対応の立体４目並べAI
        """
        self.player_num = player
        
        # 置換表の統計をリセット
        self.tt_hits = 0
        self.tt_queries = 0
        
        # ビットボードに変換
        black_board, white_board = self._convert_to_bitboard(board)
        
        # 有効な手を取得
        valid_moves = self._get_valid_moves_bb(black_board, white_board)
        if not valid_moves:
            return (0, 0)
        
        # Minimax + Alpha-Beta探索で最適手を決定（置換表対応版）
        _, best_move = self._alpha_beta_with_tt(black_board, white_board, self.max_depth, 
                                              -math.inf, math.inf, True, player)
        
        # 置換表のヒット率を出力（デバッグ用）
        if self.tt_queries > 0:
            hit_rate = (self.tt_hits / self.tt_queries) * 100
            # print(f"置換表ヒット率: {hit_rate:.1f}% ({self.tt_hits}/{self.tt_queries})")
        
        if best_move is None:
            return valid_moves[0]
        
        return best_move
    
    def _alpha_beta_with_tt(self, black_board: int, white_board: int, depth: int, 
                           alpha: float, beta: float, maximizing_player: bool, 
                           current_player: int) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        置換表を利用したAlpha-Betaプルーニング付きのMinimax探索
        
        高速化のポイント：
        1. 探索開始前に置換表をチェック
        2. 過去の結果があれば再計算をスキップ
        3. 探索終了後に結果を置換表に保存
        4. ゾブリストハッシュで盤面を高速識別
        """
        # ===== Step 1: ゾブリストハッシュを計算 =====
        hash_key = self._compute_zobrist_hash(black_board, white_board)
        original_alpha = alpha  # 置換表保存用に元のalpha値を保持
        
        # ===== Step 2: 置換表をチェック =====
        found, tt_score, tt_best_move = self._lookup_transposition_table(hash_key, depth, alpha, beta)
        if found:
            return tt_score, tt_best_move
        
        # ===== Step 3: 終了条件のチェック =====
        if depth == 0 or self._is_terminal_bb(black_board, white_board):
            score = self._evaluate_board_bb(black_board, white_board)
            # 葉ノードの結果も置換表に保存
            self._store_transposition_table(hash_key, depth, score, original_alpha, beta, None)
            return score, None
        
        # ===== Step 4: 有効手の取得と手の並び替え =====
        valid_moves = self._get_valid_moves_bb(black_board, white_board)
        if not valid_moves:
            score = self._evaluate_board_bb(black_board, white_board)
            self._store_transposition_table(hash_key, depth, score, original_alpha, beta, None)
            return score, None
        
        # 置換表から得た最善手を最初に試す（手の並び替えで高速化）
        if tt_best_move and tt_best_move in valid_moves:
            valid_moves.remove(tt_best_move)
            valid_moves.insert(0, tt_best_move)
        
        best_move = None
        
        # ===== Step 5: Minimax探索の実行 =====
        if maximizing_player:
            max_eval = -math.inf
            for move in valid_moves:
                x, y = move
                new_black, new_white, z = self._make_move_bb(black_board, white_board, x, y, current_player)
                if z == -1:
                    continue
                
                eval_score, _ = self._alpha_beta_with_tt(new_black, new_white, depth - 1, alpha, beta, False, 
                                                       2 if current_player == 1 else 1)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-Betaプルーニング
            
            # ===== Step 6: 結果を置換表に保存 =====
            self._store_transposition_table(hash_key, depth, max_eval, original_alpha, beta, best_move)
            return max_eval, best_move
            
        else:
            min_eval = math.inf
            for move in valid_moves:
                x, y = move
                new_black, new_white, z = self._make_move_bb(black_board, white_board, x, y, current_player)
                if z == -1:
                    continue
                
                eval_score, _ = self._alpha_beta_with_tt(new_black, new_white, depth - 1, alpha, beta, True, 
                                                       2 if current_player == 1 else 1)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-Betaプルーニング
            
            # ===== Step 6: 結果を置換表に保存 =====
            self._store_transposition_table(hash_key, depth, min_eval, original_alpha, beta, best_move)
            return min_eval, best_move
    
    # ===== 以下、元のコードの関数群（変更なし） =====
    
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
        if self._check_win_bb(black_board) or self._check_win_bb(white_board):
            return True
        return len(self._get_valid_moves_bb(black_board, white_board)) == 0
    
    def _check_win_bb(self, board: int) -> bool:
        """ビットボード版勝利判定"""
        for pattern in self.win_patterns:
            if (board & pattern) == pattern:
                return True
        return False
    
    def _generate_win_patterns(self) -> List[int]:
        """4つ並びの勝利パターンを事前計算"""
        patterns = []
        
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
        
        if self._check_win_bb(player_board):
            return 1000.0
        if self._check_win_bb(opponent_board):
            return -1000.0
        
        player_score = self._evaluate_threats_bb(player_board, opponent_board)
        opponent_score = self._evaluate_threats_bb(opponent_board, player_board)
        
        return player_score - opponent_score
    
    def _evaluate_threats_bb(self, my_board: int, opp_board: int) -> float:
        """ビットボード版脅威評価"""
        score = 0.0
        
        for pattern in self.win_patterns:
            my_bits = my_board & pattern
            opp_bits = opp_board & pattern
            
            if opp_bits:
                continue
            
            count = bin(my_bits).count('1')
            
            if count == 3:
                score += 50.0
            elif count == 2:
                score += 10.0  
            elif count == 1:
                score += 1.0
        
        return score
    
    # 互換性のための関数群
    def get_valid_moves(self, board: List[List[List[int]]]) -> List[Tuple[int, int]]:
        """互換性のための関数"""
        black_board, white_board = self._convert_to_bitboard(board)
        return self._get_valid_moves_bb(black_board, white_board)
    
    def alpha_beta(self, board: List[List[List[int]]], depth: int, alpha: float, beta: float, 
                   maximizing_player: bool, current_player: int) -> Tuple[float, Optional[Tuple[int, int]]]:
        """互換性のための関数"""
        black_board, white_board = self._convert_to_bitboard(board)
        return self._alpha_beta_with_tt(black_board, white_board, depth, alpha, beta, maximizing_player, current_player)