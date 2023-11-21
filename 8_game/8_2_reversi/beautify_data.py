from typing import *

import os
from collections import Counter as multiset

from tensorflow.keras.models import load_model
from tqdm import tqdm

from game import State
from game import D as positional_map
from pv_mcts import pv_mcts_action

def load_models() -> List[Callable[[State], int]]:
    model_path: List[str] = [os.path.join('./models/', fn) for fn in os.listdir('./models/') if fn.endswith('.h5')]
    models = [load_model(path) for path in model_path]
    return [pv_mcts_action(model, 1.0) for model in models]

def placed_position(state: State) -> int:
    x, y = state.slider_info
    return positional_map[x * y]

def heatmap(funcs: Callable[[State], int], state: Union[State, List[State]], count: int = 1, use_tqdm: bool = False) -> List[int]:
    states: List[State] = [state] if isinstance(state, State) else state
    result: List[int] = []
    if use_tqdm:
        for func, state in tqdm(((f, s) for f in funcs for s in states for _ in range(count)), total = len(funcs) * len(states) * count):
            result.append(placed_position(state.next(func(state))))
    else:
        result = [placed_position(state.next(func(state))) for func in funcs for state in states for _ in range(count)]
    counters: Counter[int] = multiset(result)
    return [counters[i] for i in range(36)]

def heatmap_to_latex(data: List[int], state: Optional[State] = None) -> str:
    assert len(data) == 36
    M = max(data)
    command: List[str] = [r'\begin{tikzpicture}[fill=yellow,line width=0pt,scale=2]']
    for i, (data, (y, x)) in enumerate(zip(data, ((y, x) for y in range(6) for x in range(6)))):
        if state is not None and (state.pieces[i] or state.enemy_pieces[i]):
            assert data == 0
            whose = 'boardblack' if state.pieces[i] else 'boardwhite'
            command.append(f'\\fill[fill={whose}] ({x+0.5:.1f}, {5.5-y:.1f}) circle (0.45);')
        else:
            command.append(f'\\fill[opacity={(data/M) ** 0.7:.6f}] ({x}, {6-y}) rectangle ({x+1}, {5-y});')
            command.append(f'\\node at ({x+0.5:.1f}, {5.5-y:.1f}) {{${data}$}};')
    if state is not None:
        command.append(f'\\node at (3, -0.5) {{Slider: ${state.slider_info}$}};')
    command.append(r'\end{tikzpicture}')
    return '\n'.join(command)

def main() -> None:
    functions = load_models()

    # play with functions
    r = heatmap(functions, [State(slider_info=(i, j)) for i in range(1, 10) for j in range(1, 10)], use_tqdm=True)
    # This code will result in something like:
    # [3, 5, 7, 10, 5, 9, 11, 26, 71, 32, 17, 3, 5, 87, 405, 114, 34, 16, 11, 24, 148, 52, 37, 2, 9, 20, 23, 4, 6, 1, 3, 5, 3, 6, 1, 0]
    print(r)
    with open('output.tex', 'w') as f:
        f.write(heatmap_to_latex(r) + '\n')

if __name__ == '__main__':
    main()
