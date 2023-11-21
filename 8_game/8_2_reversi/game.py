
from typing import *

import random

# L = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]
D = [ord(x) - 64 for x in '@@ABCDEFGHI@J@KLM@N@OP@@QR@ST@U@V@@WX@@@Y@Z@@[@@\\]@@@@^@_@@@@@@`a@@@@@@@b@@@@@@@@c']
L = [[ord(x) - 64 for x in r] for r in 'BIPW,AHOV],@GNU\\c,FMT[b,LSZa,CHMR,DINSX,EJOTY^,KPUZ_,QV[`'.split(',')]

class State:
    def __init__(
        self,
        pieces: Optional[List[int]] = None,
        enemy_pieces: Optional[List[int]] = None,
        slider_info: Optional[Tuple[int, int]] = None,
        depth: int = 0,
        game_end_state: Optional[Tuple[bool, bool]] = None,
    ) -> None:
        self.pieces = [0] * 36 if pieces is None else pieces
        self.enemy_pieces = [0] * 36 if enemy_pieces is None else enemy_pieces
        self.slider_info = (random.randint(1, 9), random.randint(1, 9)) if slider_info is None else slider_info
        self.depth = depth
        self.done: bool
        self.win: bool
        if game_end_state is None:
            if pieces is None and enemy_pieces is None:
                self.done = False
                self.win = False
            else:
                raise NotImplementedError('please implement the remaining done part')
        else:
            self.done, self.win = game_end_state

    def is_lose(self) -> bool:
        return not self.win

    def is_done(self) -> bool:
        return self.done

    def fetch_for_dl(self) -> List[List[int]]:
        slider_state = [0] * 36
        slider_state[self.slider_info[0] - 1] = 1
        slider_state[self.slider_info[1] + 26] = 1
        return [self.pieces.copy(), self.enemy_pieces.copy(), slider_state]

    @staticmethod
    def __aux_check__(board: List[int]) -> bool:
        def check(indices: Iterable[int]) -> bool:
            r: int = sum(board[x] << i for i, x in enumerate(indices))
            r = r & (r >> 1)
            return (r & (r >> 1) & (r >> 2)) != 0
        return any(check(range(i, i+6)) for i in range(0, 36, 6)) \
            or any(check(range(i, 36, 6)) for i in range(6)) \
            or any(check(row) for row in L)

    def next(self, action: int) -> 'State':
        state: 'State' = State(self.pieces.copy(), self.enemy_pieces.copy(), self.slider_info, self.depth + 1, (self.done, self.win))
        idx = D[(action + 1) * state.slider_info[1] if action < 9 else (action - 8) * state.slider_info[0]]
        if state.pieces[idx] == 1 or state.enemy_pieces[idx] == 1:
            raise ValueError('not a possible action')
        state.pieces[idx] = 1
        if action < 9:
            state.slider_info = (action + 1, state.slider_info[1])
        else:
            state.slider_info = (state.slider_info[0], action - 8)
        state.pieces, state.enemy_pieces = state.enemy_pieces, state.pieces
        if State.__aux_check__(state.enemy_pieces):
            state.done = True
            state.win = False
        elif not state.legal_actions():
            state.done = True
            state.win = False
        return state

    def legal_actions(self) -> List[int]:
        if self.done:
            return []
        res: List[int] = []
        for c, (x, y) in enumerate([self.slider_info, (self.slider_info[1], self.slider_info[0])]):
            for i in range(1, 10):
                if i != x:
                    idx = D[y * i]
                    if self.pieces[idx] != 1 and self.enemy_pieces[idx] != 1:
                        res.append(c * 9 + i - 1)
        return res

    def is_first_player(self) -> bool:
        return self.depth % 2 == 0

    def __str__(self):
        ox = ' xoo' if self.is_first_player() else ' oxx'
        return ''.join(
            ox[(int(self.pieces[i] == 1) << 1) | int(self.enemy_pieces[i] == 1)]
                + ('\n' if i % 6 == 5 else '')
            for i in range(36)
        ) + '%d %d' % self.slider_info

def random_action(state):
    legal_actions = state.legal_actions()
    return random.choice(legal_actions)

if __name__ == '__main__':
    state = State()

    while not state.is_done():
        state = state.next(random_action(state))

        print(state)
        print()
