# ====================
# 학습 사이클 실행
# ====================

# 패키지 임포트
from typing import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('absl').disabled = True

import tensorflow as tf

from dual_network import dual_network
from self_play import self_play, SP_GAME_COUNT
from train_network import train_network
from evaluate_network import evaluate_network

import time
class CheckTime():
    def __init__(self) -> None:
        self.t = time.time()

    def checkpoint(self) -> None:
        t = time.time()
        d = t - self.t
        if d < 1:
            print(f'Elapsed {d * 1000:.2f}ms')
        elif d < 60:
            s = int(d)
            print(f'Elapsed {s}s {(d - s) * 1000:.2f}ms')
        else:
            d = int(d)
            u: List[str] = []
            h = d // 3600
            m = d % 3600 // 60
            s = d % 60
            d //= 86400
            if d > 0:
                u.append(f'{d}d')
            if h > 0:
                u.append(f'{h}h')
            if m > 0:
                u.append(f'{m}m')
            if s > 0:
                u.append(f'{s}s')
            print(f'Elapsed {" ".join(u)}')
        self.t = t

def main():
    # 듀얼 네트워크 생성
    dual_network()

    for i in range(10):
        temp = CheckTime()
        print('Train', i, '====================')
        # 셀프 플레이 파트
        # with tf.device('/cpu:0'):
        #     self_play(core=6)

        # 행렬을 어느 정도 다루고 난 다음에 multiprocessing을 시행하려 하면, data copy overhead가 훨씬 커서 프로그램이 죽습니다.
        # 우리는 죽지 않는다는 걸 알지만 tf가 이러한 행동을 막는 거 같아요
        # 새 프로그램을 여는 방향으로 수정했습니다. self_play.py의 __name__ == '__main__' behavior도 이에 맞추어 수정했습니다.
        if os.system(f'bash -c "source venv/bin/activate && echo {SP_GAME_COUNT} 6 | python3 self_play.py"'):
            exit(-1)
        temp.checkpoint()

        # 파라미터 변경 파트
        train_network()
        temp.checkpoint()

        # 신규 파라미터 평가 파트
        evaluate_network()
        temp.checkpoint()

if __name__ == '__main__':
    main()
