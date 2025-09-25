# ファイル拡張子	.py
# 必須関数	"必須関数 get_move(
#     board: list[list[list[int]]],
#     player: int,
#     last_move: tuple[int, int, int]
# ) -> tuple[int, int]
# "
# 戻り値	(x, y) のタプル（0〜3 の範囲）
# 利用可能ライブラリ	Python標準ライブラリのみ（）
# 禁止ライブラリ	os, sys, subprocess, socket, requests, urllib, http, asyncio, threading, multiprocessing, など
# 禁止関数	open, eval, exec, compile, __import__, system, popen
# Pythonバージョン	サーバは Python 3.9 互換 で実行（match文など3.10以降専用構文は不可）
# 実行制限	メモリ最大 約1GB、CPU時間 約3秒、1手あたり待ち時間上限 30秒

# https://docs.python.org/ja/3.9/library/index.html Python 3.9 標準ライブラリドキュメント(必見だべや！）

# https://qiita.com/a_uchida/items/bec46c20fd2965c6e1a0 ゾブリストハッシュってな～に？
# http://www.amy.hi-ho.ne.jp/okuhara/howtoj.htm 置換表って単純だけど便利だよね。
# ゾブリストハッシュ: 盤面を数値に変換する計算方法
# 置換表: ゾブリストハッシュのキーとして探索結果を保存するテーブル

# bhttps://speakerdeck.com/antenna_three/bitutobodojie-shuo?slide=55 BitBoardって何なの奥様？
# https://qiita.com/zawawahoge/items/8bbd4c2319e7f7746266 ビットカウント最高効率だヒャアッ！


from typing import Optional, Dict
# from local_driver import Alg3D, Board # ローカル検証用
from framework import Alg3D, Board # 本番用
import math
import random
from dataclasses import dataclass

"""
高速ビットカウント関数
bin().count('1')より3-5倍高速なビットポピュレーションカウント実装
"""
def popcount(x):
    '''xの立っているビット数をカウントする関数
    (xは64bit整数)'''

    # 2bitごとの組に分け、立っているビット数を2bitで表現する
    x = x - ((x >> 1) & 0x5555555555555555)

    # 4bit整数に 上位2bit + 下位2bit を計算した値を入れる
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)

    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f # 8bitごと
    x = x + (x >> 8) # 16bitごと
    x = x + (x >> 16) # 32bitごと
    x = x + (x >> 32) # 64bitごと = 全部の合計
    return x & 0x0000007f

"""
置換表（Transposition Table）のエントリを表すクラス
- hash_key: ゾブリストハッシュ値（ハッシュ衝突検出用）
- depth: この評価値を計算した際の探索深度
- score: 盤面の評価値
- flag: 評価値のタイプ（EXACT, LOWERBOUND, UPPERBOUND）
- best_move: この盤面での最善手
"""
@dataclass
class TranspositionEntry:
    hash_key: int
    depth: int 
    score: float
    flag: str  # "EXACT", "LOWERBOUND", "UPPERBOUND"
    best_move: Optional[tuple[int, int]]

"""
立体四目並べAI - Bitboard + 置換表 + ゾブリストハッシュ + 高速ビットカウント実装
主要技術: Bitboard, Alpha-Beta pruning, 置換表, ゾブリストハッシュ, 高速popcount
"""
class MyAI(Alg3D):
    
    """
    AIの初期化 - 各種データ構造とハッシュテーブルの準備
    """
    def __init__(self):
        self.max_depth = 4  # 探索深度
        self.player_num = None  # 自分のプレイヤー番号
        
        # 勝利パターンを事前計算（ビットマスク）
        self.win_patterns = self._generate_win_patterns()
        
        # ゾブリストハッシュテーブルの初期化
        self.zobrist_table = self._initialize_zobrist_table()
        
        # 置換表の初期化（メモリ制限対応）
        self.max_table_size = 1000000  # 最大100万エントリ（約100MB）
        self.transposition_table: Dict[int, TranspositionEntry] = {}
        
        # 統計情報（デバッグ用）
        self.tt_hits = 0
        self.tt_queries = 0
        
    """
    ゾブリストハッシュテーブルを初期化
    64位置 × 2プレイヤー分のランダム値を生成
    """
    def _initialize_zobrist_table(self) -> list[list[int]]:
        random.seed(42)  # 再現性のため固定シード使用
        zobrist_table = []
        
        for position in range(64):  # 4x4x4 = 64位置
            player_values = []
            for player in range(2):  # プレイヤー0（黒）、プレイヤー1（白）
                random_value = random.randint(0, (1 << 63) - 1)
                player_values.append(random_value)
            zobrist_table.append(player_values)
        
        return zobrist_table
    
    """
    ビットボードからゾブリストハッシュ値を計算
    黒石・白石の全位置に対応するゾブリスト値をXORで合成
    """
    def _compute_zobrist_hash(self, black_board: int, white_board: int) -> int:
        hash_value = 0
        
        # 黒石のハッシュ値計算
        temp_black = black_board
        while temp_black:
            position = (temp_black & -temp_black).bit_length() - 1
            hash_value ^= self.zobrist_table[position][0]
            temp_black &= temp_black - 1
        
        # 白石のハッシュ値計算
        temp_white = white_board
        while temp_white:
            position = (temp_white & -temp_white).bit_length() - 1
            hash_value ^= self.zobrist_table[position][1]
            temp_white &= temp_white - 1
            
        return hash_value
    
    """
    置換表のサイズ管理（メモリ制限対応）
    テーブルサイズが上限を超えた場合、古いエントリを削除
    """
    def _manage_table_size(self):
        if len(self.transposition_table) > self.max_table_size:
            # テーブルサイズを半分にする（簡単な実装）
            items = list(self.transposition_table.items())
            keep_count = len(items) // 2
            self.transposition_table = dict(items[:keep_count])
    
    """
    置換表から過去の探索結果を検索
    同じ盤面があれば再計算をスキップし、Alpha-Betaウィンドウとの整合性をチェック
    """
    def _lookup_transposition_table(self, hash_key: int, depth: int, alpha: float, beta: float) -> tuple[bool, float, Optional[tuple[int, int]]]:
        self.tt_queries += 1
        
        if hash_key not in self.transposition_table:
            return False, 0.0, None
        
        entry = self.transposition_table[hash_key]
        
        # ハッシュ衝突検証
        if entry.hash_key != hash_key:
            return False, 0.0, None
        
        # 探索深度が不十分な場合は使用しない
        if entry.depth < depth:
            return False, 0.0, None
        
        # 評価値タイプに応じた利用可能性判定
        if entry.flag == "EXACT":
            self.tt_hits += 1
            return True, entry.score, entry.best_move
        elif entry.flag == "LOWERBOUND" and entry.score >= beta:
            self.tt_hits += 1
            return True, entry.score, entry.best_move
        elif entry.flag == "UPPERBOUND" and entry.score <= alpha:
            self.tt_hits += 1
            return True, entry.score, entry.best_move
        
        # 使用できないが、best_moveの情報は有用
        return False, 0.0, entry.best_move
    
    """
    置換表に探索結果を保存
    評価値のタイプ（EXACT/LOWERBOUND/UPPERBOUND）を判定して保存
    """
    def _store_transposition_table(self, hash_key: int, depth: int, score: float, 
                                  alpha: float, beta: float, best_move: Optional[tuple[int, int]]):
        # メモリ使用量管理
        self._manage_table_size()
        
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

    """
    メインのAI思考ルーチン
    置換表+ゾブリストハッシュ対応の立体４目並べAI
    """
    def get_move(
        self,
        board: list[list[list[int]]], # 盤面情報
        player: int, # 先手(黒):1 後手(白):2
        last_move: tuple[int, int, int] # 直前に置かれた場所(x, y, z)
    ) -> tuple[int, int]:
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
        # if self.tt_queries > 0:
        #     hit_rate = (self.tt_hits / self.tt_queries) * 100
        #     print(f"置換表ヒット率: {hit_rate:.1f}% ({self.tt_hits}/{self.tt_queries})")
        
        if best_move is None:
            return valid_moves[0]
        
        return best_move
    
    """
    置換表を利用したAlpha-Betaプルーニング付きのMinimax探索
    ゾブリストハッシュによる高速盤面識別と置換表による重複計算削減
    """
    def _alpha_beta_with_tt(self, black_board: int, white_board: int, depth: int, 
                           alpha: float, beta: float, maximizing_player: bool, 
                           current_player: int) -> tuple[float, Optional[tuple[int, int]]]:
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
    
    # ===== ビットボード関連の基本操作 =====
    
    """
    3次元リストをビットボードに変換
    4x4x4の盤面を64ビット整数2つ（黒・白）で表現
    """
    def _convert_to_bitboard(self, board: list[list[list[int]]]) -> tuple[int, int]:
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
    
    """
    ビットボードから有効な手を取得
    各(x,y)位置の最上層（z=3）が空いているかをチェック
    """
    def _get_valid_moves_bb(self, black_board: int, white_board: int) -> list[tuple[int, int]]:
        valid_moves = []
        occupied = black_board | white_board
        
        for x in range(4):
            for y in range(4):
                # 一番上の層（z=3）が空かチェック
                top_bit_pos = 3 * 16 + y * 4 + x
                if not (occupied & (1 << top_bit_pos)):
                    valid_moves.append((x, y))
        
        return valid_moves
    
    """
    ビットボードに手を打ち、新しいボードとz座標を返す
    重力により下から順に石が積み重なる
    """
    def _make_move_bb(self, black_board: int, white_board: int, x: int, y: int, 
                     player: int) -> tuple[int, int, int]:
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
    
    """
    ビットボード版ゲーム終了判定
    勝者がいるか、盤面が満杯かをチェック
    """
    def _is_terminal_bb(self, black_board: int, white_board: int) -> bool:
        if self._check_win_bb(black_board) or self._check_win_bb(white_board):
            return True
        return len(self._get_valid_moves_bb(black_board, white_board)) == 0
    
    """
    ビットボード版勝利判定
    事前計算した勝利パターンとのビット演算でO(1)判定
    """
    def _check_win_bb(self, board: int) -> bool:
        for pattern in self.win_patterns:
            if (board & pattern) == pattern:
                return True
        return False
    
    """
    4つ並びの勝利パターンを事前計算
    13方向（X/Y/Z軸、各種対角線）の全パターンをビットマスクで生成
    """
    def _generate_win_patterns(self) -> list[int]:
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
    
    # ===== 盤面評価関数 =====
    
    """
    ビットボード版盤面評価関数
    勝利状態の特別評価 + 連続性（脅威）の評価
    """
    def _evaluate_board_bb(self, black_board: int, white_board: int) -> float:
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
    
    """
    ビットボード版脅威評価【高速ビットカウント版】
    勝利パターンから自分の石の連続性を評価（3つ並び50点、2つ並び10点、1つ1点）
    従来のbin().count('1')より3-5倍高速なpopcount()を使用
    """
    def _evaluate_threats_bb(self, my_board: int, opp_board: int) -> float:
        score = 0.0
        
        for pattern in self.win_patterns:
            my_bits = my_board & pattern
            opp_bits = opp_board & pattern
            
            if opp_bits:
                continue
            
            # ===== 高速化：popcountを使用 =====
            count = popcount(my_bits)  # 従来のbin(my_bits).count('1')より3-5倍高速！
            
            if count == 3:
                score += 50.0
            elif count == 2:
                score += 10.0  
            elif count == 1:
                score += 1.0
        
        return score
    
    # ===== 互換性のための関数群 =====
    
    """
    互換性のための関数 - 有効手取得
    """
    def get_valid_moves(self, board: list[list[list[int]]]) -> list[tuple[int, int]]:
        black_board, white_board = self._convert_to_bitboard(board)
        return self._get_valid_moves_bb(black_board, white_board)
    
    """
    互換性のための関数 - Alpha-Beta探索
    """
    def alpha_beta(self, board: list[list[list[int]]], depth: int, alpha: float, beta: float, 
                   maximizing_player: bool, current_player: int) -> tuple[float, Optional[tuple[int, int]]]:
        black_board, white_board = self._convert_to_bitboard(board)
        return self._alpha_beta_with_tt(black_board, white_board, depth, alpha, beta, maximizing_player, current_player)