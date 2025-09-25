from typing import List, Tuple
from local_driver import Alg3D, Board # ローカル検証用
#from framework import Alg3D, Board # 本番用

class MyAI(Alg3D):
    def get_move(
        self,
        board: List[List[List[int]]], # 盤面情報
        player: int, # 先手(黒):1 後手(白):2
        last_move: Tuple[int, int, int] # 直前に置かれた場所(x, y, z)
    ) -> Tuple[int, int]:
        # 動作確認用：最初に見つかる空きマス（値が0）に置く
        # for x in range(len(board)):
        #     for y in range(len(board[x])):
        #         for z in range(len(board[x][y])):
        #             if board[x][y][z] == 0:  # 空きマスを発見
        #                 return (x, y)  # (x, y)座標を返す
        
        # 空きマスがない場合（通常は発生しないはず）
        return (2, 4)