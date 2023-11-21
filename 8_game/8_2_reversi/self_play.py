# ====================
# 셀프 플레이 파트
# ====================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('absl').disabled = True

# 패키지 임포트
from game import State
from pv_mcts import pv_mcts_scores
from dual_network import DN_OUTPUT_SIZE
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np
import pickle

import multiprocessing

from tqdm import tqdm

# 파라미터 준비
SP_GAME_COUNT = 500  # 셀프 플레이를 수행할 게임 수(오리지널: 25,000)
SP_TEMPERATURE = 1.0  # 볼츠만 분포의 온도 파라미터


# 선 수 플레이어 가치
def first_player_value(ended_state: State):
    # 1: 선 수 플레이어 승리, -1: 선 수 플레이어 패배, 0: 무승부
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0


# 학습 데이터 저장
def write_data(history):
    now = datetime.now()
    os.makedirs('./data/', exist_ok=True)  # 폴더가 없는 경우에는 생성
    path = './data/{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    with open(path, mode='wb') as f:
        pickle.dump(history, f)


# 1 게임 실행
def play(model):
    # 학습 데이터
    history = []

    # 상태 생성
    state = State()

    while not state.is_done():
        # 게임 종료 시
        # if state.is_done():
        #     break

        # 합법적인 수의 확률 분포 얻기
        scores = pv_mcts_scores(model, state, SP_TEMPERATURE)

        # 학습 데이터에 상태와 정책 추가
        policies = [0] * DN_OUTPUT_SIZE
        legal_actions = state.legal_actions()
        assert legal_actions
        for action, policy in zip(legal_actions, scores):
            policies[action] = policy
        history.append([state.fetch_for_dl(), policies, None])

        # 행동 얻기
        action = np.random.choice(legal_actions, p=scores)

        # 다음 상태 얻기
        state = state.next(action)

    # 학습 데이터에 가치 추가
    value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = value
        value = -value
    return history


def _play_wrapper(index, model):
    data = play(model)
    try:
        timestamp = datetime.now().strftime('%H:%M:%S')
        with open('temp.out', 'a') as f:
            f.write(f'[{timestamp}] [Pid {os.getpid()}] data {index} complete\n')
    except:
        pass
    return data

# 셀프 플레이
def self_play(core: int = 1, count: int = SP_GAME_COUNT):

    # 베스트 플레이어 모델 로드
    model = load_model('./model/best.h5', compile=False)

    # 여러 차례 게임 실행
    if core > 1:
        with open('temp.out', 'w'):
            pass
        history_list = []
        BATCH = core * 3
        for i in range(0, count, BATCH):
            with multiprocessing.get_context('spawn').Pool(core) as p:
                history_list.extend(p.starmap(_play_wrapper, enumerate([model] * (min(i + BATCH, count) - i), start=i)))

        history = [y for x in history_list for y in x]
    else:
        # 학습 데이터
        history = []

        for _ in tqdm(range(count)):
            # 1ゲームの実行
            h = play(model)
            history.extend(h)

            # 출력
            # print('\rSelfPlay {}/{}'.format(i + 1, SP_GAME_COUNT), end='')
        print('')

    # 학습 데이터 저장
    write_data(history)

    # 모델 파기
    K.clear_session()
    del model

def main():
    count, core = map(int, input().split())
    self_play(core, count)

# 동작 확인
if __name__ == '__main__':
    main()
