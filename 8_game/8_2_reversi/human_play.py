# ====================
# 사람과 AI의 대전
# ====================

# 패키지 임포트
from game import State
from pv_mcts import pv_mcts_action
from tensorflow.keras.models import load_model
from pathlib import Path
from threading import Thread
import tkinter as tk

# 베스트 플레이어 모델 로드
model = load_model('./model/best.h5')


# 게임 UI 정의
class GameUI(tk.Frame):
    # 초기화
    def __init__(self, master=None, model=None):
        tk.Frame.__init__(self, master)
        self.master.title('Numbers')

        # 게임 상태 생성
        self.state = State()

        # PV MCTS를 활용한 행동을 선택하는 함수 생성
        self.next_action = pv_mcts_action(model, 0.0)

        # 캔버스 생성
        self.c = tk.Canvas(self, width=360, height=480, highlightthickness=0)
        self.c.bind('<Button-1>', self.turn_of_human)
        self.c.pack()

        # 화면 갱신
        self.on_draw()

    # 사람의 턴
    def turn_of_human(self, event):
        # 게임 종료 시
        if self.state.is_done():
            self.state = State()
            self.on_draw()
            return

        # 선 수가 아닌 경우
        if not self.state.is_first_player():
            return

        # 클릭 위치를 행동으로 변환
        action: int
        if 45 <= event.x < 315:
            if 385 <= event.y < 415:
                action = (event.x - 45) // 30
            elif 425 <= event.y < 455:
                action = 9 + (event.x - 45) // 30
            else:
                return

        # 합법적인 수가 아닌 경우
        legal_actions = self.state.legal_actions()
        if action not in legal_actions:
            return

        # 다음 상태 얻기
        self.state = self.state.next(action)
        self.on_draw()

        # AI의 턴
        self.master.after(1, self.turn_of_ai)

    # AI의 턴
    def turn_of_ai(self):
        # 게임 종료 시
        if self.state.is_done():
            return

        # 행동 얻기
        action = self.next_action(self.state)

        # 다음 상태 얻기
        self.state = self.state.next(action)
        self.on_draw()

    # 돌 그리기
    def draw_piece(self, index, first_player):
        x = (index % 6) * 60 + 5
        y = index // 6 * 60 + 5
        if first_player:
            self.c.create_oval(x, y, x + 50, y + 50, width=1.0, outline='#000000', fill='#009900')
        else:
            self.c.create_oval(x, y, x + 50, y + 50, width=1.0, outline='#000000', fill='#cc00cc')

    # 화면 갱신
    def on_draw(self):
        self.c.delete('all')
        self.c.create_rectangle(0, 0, 360, 480, width=0.0, fill='black')
        for i in range(36):
            if self.state.pieces[i] == 1:
                self.draw_piece(i, self.state.is_first_player())
            if self.state.enemy_pieces[i] == 1:
                self.draw_piece(i, not self.state.is_first_player())

        self.c.create_line(0, 360, 360, 360, width=1.0, fill='white')
        self.c.create_rectangle(45, 385, 315, 415, width=0.0, fill='#333333')
        self.c.create_rectangle(45, 425, 315, 455, width=0.0, fill='#333333')

        x,y = self.state.slider_info
        self.c.create_rectangle(15 + x * 30, 385, 45 + x * 30, 415, width=0.0, fill='#000099')
        self.c.create_rectangle(15 + y * 30, 425, 45 + y * 30, 455, width=0.0, fill='#000099')
        D = list({x * y for x in range(1, 10) for y in range(1, 10)})
        D.sort()
        for i,x in enumerate(D):
            self.c.create_text((i % 6) * 60 + 30, (i // 6) * 60 + 30, text=f'{x}', fill='white', font=('Helvetica 15'))
        for i in range(1, 10):
            self.c.create_text(i * 30 + 30, 400, text=f'{i}', fill='white', font=('Helvetica 15'))
            self.c.create_text(i * 30 + 30, 440, text=f'{i}', fill='white', font=('Helvetica 15'))

# 게임 UI 실행
f = GameUI(model=model)
f.pack()
f.mainloop()
