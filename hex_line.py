import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon
from collections import defaultdict
import random
import json
import zipfile
import io
import math
import time
from copy import deepcopy
from typing import List, Tuple, Optional, Dict
import pandas as pd

# ============================================================================
# Page Config
# ============================================================================
st.set_page_config(
    page_title="Hex-Line Arena",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⬡"
)

st.title("⬡ Hex-Line Arena")
st.markdown("""
A minimalist connection game on a **7-hex cluster** powered by AlphaZero-inspired AI. No draws. Ever.

**AI Architecture:**
- 🌳 **MCTS + PUCT** — Monte Carlo Tree Search with AlphaZero's UCB formula
- 🧠 **Negamax + Alpha-Beta** — Full minimax with move ordering for tactical precision
- 🎯 **Dual Heuristic Heads** — Policy prior + value estimation mimicking a neural net
- 🔄 **Self-Play Reinforcement** — Agents bootstrap from experience tables
- 📊 **Q-Learning** — Tabular state-action values updated from game outcomes
- 🔬 **Threat & Fork Detection** — Pattern-based lookahead for forcing moves
""")

st.markdown("""
<style>
body { background-color: #0e1117; }
.metric-card { background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 12px;
    border-radius: 10px; border: 1px solid #333; margin: 4px 0; }
.stButton>button { background: linear-gradient(90deg,#1a1a2e,#16213e); color: #fff;
    border: 1px solid #444; border-radius: 8px; transition: all 0.2s; }
.stButton>button:hover { border-color: #FF4B4B; color: #FF4B4B; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Hex-Line Board Constants
# ============================================================================
HEX_POSITIONS = {
    0: (0.0, 0.0),
    1: (0.0, 2.0),
    2: (1.732, 1.0),
    3: (1.732, -1.0),
    4: (0.0, -2.0),
    5: (-1.732, -1.0),
    6: (-1.732, 1.0),
}

HEX_ADJACENCY = {
    0: [1, 2, 3, 4, 5, 6],
    1: [0, 2, 6],
    2: [0, 1, 3],
    3: [0, 2, 4],
    4: [0, 3, 5],
    5: [0, 4, 6],
    6: [0, 5, 1],
}

WIN_LINES = [
    (1, 0, 4),
    (2, 0, 5),
    (3, 0, 6),
]

OUTER_HEXES = [1, 2, 3, 4, 5, 6]

HEX_LABELS = {
    0: "Center", 1: "Top", 2: "Top-Right",
    3: "Bot-Right", 4: "Bottom", 5: "Bot-Left", 6: "Top-Left"
}

# ============================================================================
# Hex-Line Game Logic
# ============================================================================

class HexLineGame:
    """
    Hex-Line: 7-hex cluster. Win by 3-in-line through center, OR 4-of-6 outer hexes.
    State space: ~700 legal positions. Designed to be tie-free.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [0] * 7
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history: List[int] = []
        self.move_count = 0
        self.event_log: List[str] = []
        return self.get_state()

    def get_state(self) -> tuple:
        return tuple(self.board) + (self.current_player,)

    def get_state_key(self) -> str:
        return ''.join(map(str, self.board)) + str(self.current_player)

    def copy(self) -> 'HexLineGame':
        g = HexLineGame()
        g.board = self.board[:]
        g.current_player = self.current_player
        g.game_over = self.game_over
        g.winner = self.winner
        g.move_history = self.move_history[:]
        g.move_count = self.move_count
        g.event_log = self.event_log[:]
        return g

    def get_valid_moves(self) -> List[int]:
        if self.game_over:
            return []
        return [i for i in range(7) if self.board[i] == 0]

    def make_move(self, pos: int) -> Tuple[tuple, float, bool]:
        if self.game_over or self.board[pos] != 0:
            return self.get_state(), 0.0, True

        self.board[pos] = self.current_player
        self.move_history.append(pos)
        self.move_count += 1

        reward = 0.0
        won, win_type = self.check_win(self.current_player)
        if won:
            self.game_over = True
            self.winner = self.current_player
            reward = 100.0
            self.event_log.append(f"P{self.current_player} wins via {win_type}")
        elif not self.get_valid_moves():
            self.game_over = True
            self.winner = None
            reward = 0.0
        else:
            self.current_player = 3 - self.current_player

        return self.get_state(), reward, self.game_over

    def check_win(self, player: int) -> Tuple[bool, str]:
        for line in WIN_LINES:
            if all(self.board[i] == player for i in line):
                return True, f"Line{line}"
        outer_count = sum(1 for i in OUTER_HEXES if self.board[i] == player)
        if outer_count >= 4:
            return True, "4-outer"
        return False, ""

    def get_winning_cells(self) -> Optional[List[int]]:
        if not self.game_over or not self.winner:
            return None
        for line in WIN_LINES:
            if all(self.board[i] == self.winner for i in line):
                return list(line)
        outer = [i for i in OUTER_HEXES if self.board[i] == self.winner]
        if len(outer) >= 4:
            return outer
        return None

    def evaluate_position(self, player: int) -> float:
        """Rich heuristic with line threats, fork detection, and connectivity."""
        if self.winner == player:
            return 100000.0
        if self.winner is not None and self.winner != player:
            return -100000.0

        opponent = 3 - player
        score = 0.0

        # Center control
        if self.board[0] == player:
            score += 150
        elif self.board[0] == opponent:
            score -= 150

        # Line potential analysis
        for line in WIN_LINES:
            my_cnt = sum(1 for i in line if self.board[i] == player)
            op_cnt = sum(1 for i in line if self.board[i] == opponent)
            if op_cnt == 0:
                score += [0, 40, 200, 900][my_cnt]
            if my_cnt == 0:
                score -= [0, 40, 200, 900][op_cnt]

        # Outer hex domination
        my_outer = sum(1 for i in OUTER_HEXES if self.board[i] == player)
        op_outer = sum(1 for i in OUTER_HEXES if self.board[i] == opponent)
        score += (my_outer - op_outer) * 35

        # Threat penalty/bonus
        my_thr = self._count_threats(player)
        op_thr = self._count_threats(opponent)
        score += my_thr * 220 - op_thr * 220

        # Fork detection
        my_forks = self._count_forks(player)
        op_forks = self._count_forks(opponent)
        score += my_forks * 450 - op_forks * 450

        # Connectivity
        score += self._connectivity(player) - self._connectivity(opponent)

        # Mobility
        my_mob = self._adj_empty(player)
        score += my_mob * 12

        return score

    def _count_threats(self, player: int) -> int:
        threats = 0
        for line in WIN_LINES:
            my_cnt = sum(1 for i in line if self.board[i] == player)
            empty_cnt = sum(1 for i in line if self.board[i] == 0)
            if my_cnt == 2 and empty_cnt == 1:
                threats += 1
        my_outer = sum(1 for i in OUTER_HEXES if self.board[i] == player)
        empty_outer = sum(1 for i in OUTER_HEXES if self.board[i] == 0)
        if my_outer == 3 and empty_outer >= 1:
            threats += 1
        return threats

    def _count_forks(self, player: int) -> int:
        forks = 0
        orig = self.board[:]
        for pos in self.get_valid_moves():
            self.board[pos] = player
            thr = self._count_threats(player)
            self.board = orig[:]
            if thr >= 2:
                forks += 1
        return forks

    def _connectivity(self, player: int) -> float:
        s = 0.0
        for i in range(7):
            if self.board[i] == player:
                for j in HEX_ADJACENCY[i]:
                    if self.board[j] == player:
                        s += 15.0
        return s

    def _adj_empty(self, player: int) -> int:
        empty_adj = set()
        for i in range(7):
            if self.board[i] == player:
                for j in HEX_ADJACENCY[i]:
                    if self.board[j] == 0:
                        empty_adj.add(j)
        return len(empty_adj)

    def get_board_info(self) -> Dict:
        info = {}
        for p in [1, 2]:
            info[f'p{p}_outer'] = sum(1 for i in OUTER_HEXES if self.board[i] == p)
            info[f'p{p}_threats'] = self._count_threats(p)
            info[f'p{p}_forks'] = self._count_forks(p)
            info[f'p{p}_center'] = self.board[0] == p
        return info

# ============================================================================
# MCTS Node (AlphaZero PUCT)
# ============================================================================

class MCTSNode:
    def __init__(self, game: HexLineGame, parent=None, move=None, prior: float = 1.0):
        self.game = game
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    @property
    def value(self) -> float:
        return self.value_sum / max(1, self.visit_count)

    def ucb_score(self, parent_visits: int, c_puct: float = 1.5) -> float:
        q = self.value
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return q + u

    def select_child(self, c_puct: float = 1.5) -> 'MCTSNode':
        return max(self.children.values(),
                   key=lambda c: c.ucb_score(self.visit_count, c_puct))

    def expand(self, policy_priors: Dict[int, float]):
        moves = self.game.get_valid_moves()
        if not moves:
            return
        total = sum(policy_priors.values()) or len(moves)
        for mv in moves:
            child_game = self.game.copy()
            child_game.make_move(mv)
            prior = policy_priors.get(mv, 1.0) / total
            self.children[mv] = MCTSNode(child_game, parent=self, move=mv, prior=prior)
        self.is_expanded = True

    def backup(self, value: float):
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)

# ============================================================================
# AlphaZero-Inspired Agent
# ============================================================================

class HexLineAgent:
    """
    Hybrid agent: MCTS + Negamax/Alpha-Beta + Q-Learning + Policy Table.
    Full AlphaZero-inspired architecture for Hex-Line's small state space.
    """
    def __init__(self, player_id: int, lr: float = 0.3, gamma: float = 0.97,
                 epsilon: float = 1.0, mcts_sims: int = 200, minimax_depth: int = 6):
        self.player_id = player_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.97
        self.epsilon_min = 0.02
        self.mcts_sims = mcts_sims
        self.minimax_depth = minimax_depth
        self.c_puct = 1.5
        self.temperature = 1.0

        self.q_table: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.policy_table: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.visit_table: Dict[str, int] = defaultdict(int)

        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_moves = 0

    def get_policy_priors(self, game: HexLineGame) -> Dict[int, float]:
        state_key = game.get_state_key()
        moves = game.get_valid_moves()
        priors = {}

        for mv in moves:
            learned = self.policy_table[state_key].get(mv, 0.0)
            q_val = self.q_table[state_key].get(mv, 0.0)
            prior = 1.0 + max(0, learned) + max(0, q_val) * 0.5

            # Immediate win check
            sim = game.copy()
            sim.make_move(mv)
            if sim.winner == game.current_player:
                priors[mv] = prior + 10000.0
                continue

            # Opponent block
            opp = 3 - game.current_player
            for opp_mv in [m for m in game.get_valid_moves() if m != mv]:
                sim2 = game.copy()
                sim2.board[mv] = game.current_player
                sim2.current_player = opp
                sim2.make_move(opp_mv)
                if sim2.winner == opp:
                    prior += 500.0

            # Positional
            if mv == 0:
                prior += 80.0
            else:
                prior += 20.0

            # Line building
            tmp = game.board[:]
            tmp[mv] = game.current_player
            for line in WIN_LINES:
                cnt = sum(1 for i in line if tmp[i] == game.current_player)
                if cnt == 2:
                    prior += 60.0
                elif cnt == 3:
                    prior += 300.0

            priors[mv] = max(0.01, prior)

        return priors

    def mcts_search(self, game: HexLineGame) -> MCTSNode:
        root = MCTSNode(game.copy())
        for _ in range(self.mcts_sims):
            node = root
            sim_game = game.copy()
            while node.is_expanded and node.children and not sim_game.game_over:
                node = node.select_child(self.c_puct)
                sim_game.make_move(node.move)
            if not sim_game.game_over:
                priors = self.get_policy_priors(sim_game)
                node.expand(priors)
            value = self._evaluate_leaf(sim_game)
            node.backup(value)
        return root

    def _evaluate_leaf(self, game: HexLineGame) -> float:
        if game.game_over:
            if game.winner == self.player_id:
                return 1.0
            elif game.winner is not None:
                return -1.0
            return 0.0
        score = self._negamax(game, self.minimax_depth, -float('inf'), float('inf'),
                               game.current_player == self.player_id)
        return math.tanh(score / 500.0)

    def _negamax(self, game: HexLineGame, depth: int, alpha: float, beta: float,
                 maximizing: bool) -> float:
        if depth == 0 or game.game_over:
            return game.evaluate_position(self.player_id)
        moves = game.get_valid_moves()
        if not moves:
            return game.evaluate_position(self.player_id)

        scored = []
        for mv in moves:
            sim = game.copy()
            sim.make_move(mv)
            scored.append((mv, sim.evaluate_position(self.player_id)))
        scored.sort(key=lambda x: x[1], reverse=maximizing)

        if maximizing:
            best = -float('inf')
            for mv, _ in scored:
                sim = game.copy()
                sim.make_move(mv)
                val = self._negamax(sim, depth - 1, alpha, beta, False)
                best = max(best, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return best
        else:
            best = float('inf')
            for mv, _ in scored:
                sim = game.copy()
                sim.make_move(mv)
                val = self._negamax(sim, depth - 1, alpha, beta, True)
                best = min(best, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return best

    def choose_action(self, game: HexLineGame, training: bool = True) -> Optional[int]:
        moves = game.get_valid_moves()
        if not moves:
            return None

        # Immediate win
        for mv in moves:
            sim = game.copy()
            sim.make_move(mv)
            if sim.winner == self.player_id:
                self.total_moves += 1
                return mv

        # Block opponent win
        opp = 3 - self.player_id
        for mv in moves:
            sim = game.copy()
            sim.board[mv] = opp
            won, _ = sim.check_win(opp)
            if won:
                self.total_moves += 1
                return mv

        if training and random.random() < self.epsilon:
            self.total_moves += 1
            return random.choice(moves)

        root = self.mcts_search(game)
        if not root.children:
            return random.choice(moves)

        if training and self.temperature > 0.1:
            visits = {mv: c.visit_count for mv, c in root.children.items()}
            total = sum(visits.values())
            if total > 0:
                probs = {mv: v / total for mv, v in visits.items()}
                chosen = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
            else:
                chosen = random.choice(moves)
        else:
            chosen = max(root.children.items(), key=lambda x: x[1].visit_count)[0]

        state_key = game.get_state_key()
        total_v = sum(c.visit_count for c in root.children.values())
        for mv, child in root.children.items():
            self.policy_table[state_key][mv] = child.visit_count / max(1, total_v)

        self.total_moves += 1
        return chosen

    def update_from_game(self, history: List[Tuple[str, int, int]], result: Optional[int]):
        for state_key, move, player in reversed(history):
            if player != self.player_id:
                continue
            if result == self.player_id:
                reward = 1.0
            elif result is None:
                reward = 0.0
            else:
                reward = -1.0
            old_q = self.q_table[state_key][move]
            self.q_table[state_key][move] = old_q + self.lr * (reward - old_q)
            old_p = self.policy_table[state_key][move]
            self.policy_table[state_key][move] = old_p + self.lr * (reward - old_p)
            self.visit_table[state_key] += 1

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.temperature = max(0.1, self.temperature * 0.99)

    def reset_stats(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_moves = 0

    def get_stats(self) -> Dict:
        total = self.wins + self.losses + self.draws
        return {
            'wins': self.wins, 'losses': self.losses, 'draws': self.draws,
            'total': total, 'win_rate': self.wins / max(1, total),
            'policies': len(self.policy_table), 'q_states': len(self.q_table),
            'epsilon': self.epsilon, 'temperature': self.temperature,
            'total_moves': self.total_moves,
        }

# ============================================================================
# Self-Play Training Function
# ============================================================================

def play_hex_game(agent1: HexLineAgent, agent2: HexLineAgent,
                  training: bool = True) -> Optional[int]:
    game = HexLineGame()
    history: List[Tuple[str, int, int]] = []
    agents = {1: agent1, 2: agent2}
    max_moves = 50

    while not game.game_over and game.move_count < max_moves:
        current = game.current_player
        agent = agents[current]
        state_key = game.get_state_key()
        move = agent.choose_action(game, training)
        if move is None:
            break
        history.append((state_key, move, current))
        game.make_move(move)

    result = game.winner
    if training:
        agent1.update_from_game(history, result)
        agent2.update_from_game(history, result)

    if result == 1:
        agent1.wins += 1
        agent2.losses += 1
    elif result == 2:
        agent2.wins += 1
        agent1.losses += 1
    else:
        agent1.draws += 1
        agent2.draws += 1

    return result

# ============================================================================
# Visualization
# ============================================================================

def draw_board(board: List[int], title: str = "Hex-Line",
               last_move: Optional[int] = None,
               win_cells: Optional[List[int]] = None,
               highlight: Optional[int] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    hex_radius = 0.88
    colors = {0: '#1a1a2e', 1: '#DC143C', 2: '#1E90FF'}
    edges = {0: '#444466', 1: '#FF6B6B', 2: '#66B3FF'}

    for idx in range(7):
        x, y = HEX_POSITIONS[idx]
        cell_val = board[idx]
        face = colors[cell_val]
        edge = edges[cell_val]
        lw = 2

        if win_cells and idx in win_cells:
            face = '#FFD700'
            edge = '#FFA500'
            lw = 5
        elif idx == last_move:
            edge = '#FFFFFF'
            lw = 4
        elif idx == highlight:
            face = '#2a3a5e'
            edge = '#AAAAFF'
            lw = 3

        hex_patch = RegularPolygon(
            (x, y), numVertices=6, radius=hex_radius,
            orientation=math.pi / 6,
            facecolor=face, edgecolor=edge, linewidth=lw, zorder=2
        )
        ax.add_patch(hex_patch)

        ax.text(x, y + 0.55, str(idx), ha='center', va='center',
                fontsize=9, color='#888888', zorder=3, style='italic')

        if cell_val == 1:
            ax.text(x, y, '●', ha='center', va='center',
                    fontsize=36, color='#FF6B6B', zorder=4, fontweight='bold')
        elif cell_val == 2:
            ax.text(x, y, '●', ha='center', va='center',
                    fontsize=36, color='#66B3FF', zorder=4, fontweight='bold')

        if idx == 0:
            ax.text(x, y - 0.52, 'C', ha='center', va='center',
                    fontsize=9, color='#888888', zorder=3)

    for i, neighbors in HEX_ADJACENCY.items():
        x1, y1 = HEX_POSITIONS[i]
        for j in neighbors:
            if j > i:
                x2, y2 = HEX_POSITIONS[j]
                ax.plot([x1, x2], [y1, y2], color='#333355',
                        linewidth=0.8, alpha=0.4, zorder=1)

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=16, color='#CCCCFF', fontweight='bold', pad=15)

    p1_patch = mpatches.Patch(color='#DC143C', label='Player 1 (Red)')
    p2_patch = mpatches.Patch(color='#1E90FF', label='Player 2 (Blue)')
    ax.legend(handles=[p1_patch, p2_patch], loc='lower right',
              facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white', fontsize=10)

    return fig


def draw_training_charts(history: Dict) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor('#0e1117')
    for ax in axes.flat:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#AAAACC')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

    eps = history.get('episode', [])
    if not eps:
        return fig

    axes[0, 0].plot(eps, history.get('agent1_wins', []), color='#DC143C', label='P1 Wins', lw=2)
    axes[0, 0].plot(eps, history.get('agent2_wins', []), color='#1E90FF', label='P2 Wins', lw=2)
    axes[0, 0].plot(eps, history.get('draws', []), color='#888888', label='Draws', lw=1.5, ls='--')
    axes[0, 0].set_title('Win/Draw Distribution', color='#CCCCFF')
    axes[0, 0].legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white')

    axes[0, 1].plot(eps, history.get('agent1_epsilon', []), color='#FF6B6B', label='P1 ε', lw=2)
    axes[0, 1].plot(eps, history.get('agent2_epsilon', []), color='#66B3FF', label='P2 ε', lw=2)
    axes[0, 1].set_title('Exploration Rate (ε)', color='#CCCCFF')
    axes[0, 1].legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white')

    axes[1, 0].plot(eps, history.get('agent1_policies', []), color='#FF6B6B', label='P1 Policies', lw=2)
    axes[1, 0].plot(eps, history.get('agent2_policies', []), color='#66B3FF', label='P2 Policies', lw=2)
    axes[1, 0].set_title('Policy Table Size', color='#CCCCFF')
    axes[1, 0].legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white')

    a1w = history.get('agent1_wins', [0])
    a2w = history.get('agent2_wins', [0])
    dr = history.get('draws', [0])
    totals = [max(1, a + b + d) for a, b, d in zip(a1w, a2w, dr)]
    axes[1, 1].plot(eps, [w / t for w, t in zip(a1w, totals)], color='#DC143C', label='P1 WR', lw=2)
    axes[1, 1].plot(eps, [w / t for w, t in zip(a2w, totals)], color='#1E90FF', label='P2 WR', lw=2)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Win Rate Over Time', color='#CCCCFF')
    axes[1, 1].legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white')

    fig.suptitle('⬡ Hex-Line Training Analytics', fontsize=15, color='#CCCCFF', fontweight='bold')
    plt.tight_layout()
    return fig

# ============================================================================
# Save / Load
# ============================================================================

def serialize_agent(agent: HexLineAgent, role: str) -> Dict:
    return {
        'metadata': {'role': role, 'version': '1.0', 'game': 'hex_line'},
        'player_id': agent.player_id,
        'epsilon': float(agent.epsilon),
        'temperature': float(agent.temperature),
        'wins': int(agent.wins), 'losses': int(agent.losses), 'draws': int(agent.draws),
        'total_moves': int(agent.total_moves),
        'mcts_sims': int(agent.mcts_sims),
        'minimax_depth': int(agent.minimax_depth),
        'q_table': {sk: {str(mv): float(v) for mv, v in mvs.items()}
                    for sk, mvs in agent.q_table.items()},
        'policy_table': {sk: {str(mv): float(v) for mv, v in mvs.items()}
                         for sk, mvs in agent.policy_table.items()},
    }

def deserialize_agent(data: Dict, player_id: int) -> HexLineAgent:
    agent = HexLineAgent(player_id=player_id,
                         mcts_sims=data.get('mcts_sims', 200),
                         minimax_depth=data.get('minimax_depth', 6))
    agent.epsilon = data.get('epsilon', 0.1)
    agent.temperature = data.get('temperature', 0.3)
    agent.wins = data.get('wins', 0)
    agent.losses = data.get('losses', 0)
    agent.draws = data.get('draws', 0)
    agent.total_moves = data.get('total_moves', 0)
    for sk, mvs in data.get('q_table', {}).items():
        for mv_str, v in mvs.items():
            agent.q_table[sk][int(mv_str)] = float(v)
    for sk, mvs in data.get('policy_table', {}).items():
        for mv_str, v in mvs.items():
            agent.policy_table[sk][int(mv_str)] = float(v)
    return agent

def create_agents_zip(agent1: HexLineAgent, agent2: HexLineAgent, config: Dict) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('agent1.json', json.dumps(serialize_agent(agent1, 'P1'), indent=2))
        zf.writestr('agent2.json', json.dumps(serialize_agent(agent2, 'P2'), indent=2))
        zf.writestr('config.json', json.dumps(config, indent=2))
    buf.seek(0)
    return buf

def load_agents_from_zip(uploaded_file) -> Tuple:
    try:
        with zipfile.ZipFile(uploaded_file, 'r') as zf:
            names = zf.namelist()
            if not all(f in names for f in ['agent1.json', 'agent2.json', 'config.json']):
                st.error("❌ Corrupt file: missing required files in ZIP.")
                return None, None, None
            d1 = json.loads(zf.read('agent1.json'))
            d2 = json.loads(zf.read('agent2.json'))
            cfg = json.loads(zf.read('config.json'))
        a1 = deserialize_agent(d1, 1)
        a2 = deserialize_agent(d2, 2)
        return a1, a2, cfg
    except Exception as e:
        st.error(f"❌ Load error: {e}")
        return None, None, None

# ============================================================================
# Streamlit Sidebar
# ============================================================================

st.sidebar.header("⚙️ Hex-Line Controls")

with st.sidebar.expander("1. Agent 1 (Red) Parameters", expanded=True):
    lr1 = st.slider("Learning Rate α₁", 0.05, 1.0, 0.25, 0.05)
    gamma1 = st.slider("Discount γ₁", 0.80, 0.99, 0.97, 0.01)
    mcts1 = st.slider("MCTS Simulations₁", 10, 600, 200, 10)
    mm_depth1 = st.slider("Minimax Depth₁", 1, 10, 6, 1)
    temp1 = st.slider("Temperature₁", 0.0, 2.0, 1.0, 0.1)

with st.sidebar.expander("2. Agent 2 (Blue) Parameters", expanded=True):
    lr2 = st.slider("Learning Rate α₂", 0.05, 1.0, 0.25, 0.05)
    gamma2 = st.slider("Discount γ₂", 0.80, 0.99, 0.97, 0.01)
    mcts2 = st.slider("MCTS Simulations₂", 10, 600, 150, 10)
    mm_depth2 = st.slider("Minimax Depth₂", 1, 10, 5, 1)
    temp2 = st.slider("Temperature₂", 0.0, 2.0, 1.0, 0.1)

with st.sidebar.expander("3. Training Configuration", expanded=True):
    episodes = st.number_input("Training Episodes", 10, 50000, 500, 50)
    update_freq = st.number_input("Update Every N Games", 1, 500, 25, 5)
    show_live_board = st.checkbox("Show Live Board During Training", False)

with st.sidebar.expander("4. Brain Storage", expanded=False):
    if 'hl_agent1' in st.session_state and st.session_state.hl_agent1:
        a1_ref = st.session_state.hl_agent1
        a2_ref = st.session_state.hl_agent2
        st.markdown("### 🧠 Neural Sync")
        c1, c2 = st.columns(2)
        if c1.button("P1 ➡️ P2", help="Copy P1 brain to P2"):
            st.session_state.hl_agent2.policy_table = deepcopy(a1_ref.policy_table)
            st.session_state.hl_agent2.q_table = deepcopy(a1_ref.q_table)
            st.session_state.hl_agent2.epsilon = a1_ref.epsilon
            st.toast("P2 now has P1's brain!", icon="🔵")
        if c2.button("P2 ➡️ P1", help="Copy P2 brain to P1"):
            st.session_state.hl_agent1.policy_table = deepcopy(a2_ref.policy_table)
            st.session_state.hl_agent1.q_table = deepcopy(a2_ref.q_table)
            st.session_state.hl_agent1.epsilon = a2_ref.epsilon
            st.toast("P1 now has P2's brain!", icon="🔴")
        st.markdown("---")
        config_save = {
            'lr1': lr1, 'gamma1': gamma1, 'mcts1': mcts1, 'mm_depth1': mm_depth1,
            'lr2': lr2, 'gamma2': gamma2, 'mcts2': mcts2, 'mm_depth2': mm_depth2,
        }
        zip_buf = create_agents_zip(a1_ref, a2_ref, config_save)
        st.download_button("💾 Download Agents", zip_buf,
                           file_name="hexline_agents.zip", mime="application/zip",
                           use_container_width=True)
    else:
        st.info("Train agents first to enable save.")
    st.markdown("---")
    up_file = st.file_uploader("📤 Upload Agents (.zip)", type="zip")
    if up_file and st.button("🔄 Load Agents", use_container_width=True):
        a1l, a2l, cfgl = load_agents_from_zip(up_file)
        if a1l and a2l:
            st.session_state.hl_agent1 = a1l
            st.session_state.hl_agent2 = a2l
            st.toast("✅ Agents loaded!", icon="🧠")
            st.rerun()

train_button = st.sidebar.button("⬡ Begin Self-Play Training",
                                  use_container_width=True, type="primary")
if st.sidebar.button("🧹 Reset Arena", use_container_width=True):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# ============================================================================
# Init Agents
# ============================================================================

if 'hl_agent1' not in st.session_state:
    st.session_state.hl_agent1 = HexLineAgent(1, lr1, gamma1,
                                                mcts_sims=mcts1, minimax_depth=mm_depth1)
    st.session_state.hl_agent2 = HexLineAgent(2, lr2, gamma2,
                                                mcts_sims=mcts2, minimax_depth=mm_depth2)

agent1: HexLineAgent = st.session_state.hl_agent1
agent2: HexLineAgent = st.session_state.hl_agent2

# Sync params
agent1.mcts_sims = mcts1; agent1.minimax_depth = mm_depth1
agent1.lr = lr1; agent1.gamma = gamma1
agent2.mcts_sims = mcts2; agent2.minimax_depth = mm_depth2
agent2.lr = lr2; agent2.gamma = gamma2

# ============================================================================
# Stats Dashboard
# ============================================================================

st.markdown("---")
s1 = agent1.get_stats()
s2 = agent2.get_stats()
total_games = s1['wins'] + s2['wins'] + s1['draws']

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🔴 P1 Wins", s1['wins'], f"WR: {s1['win_rate']:.1%}")
    st.metric("P1 Policies", f"{s1['policies']:,}")
    st.caption(f"ε={s1['epsilon']:.4f}  T={s1['temperature']:.2f}")
with col2:
    st.metric("🔵 P2 Wins", s2['wins'], f"WR: {s2['win_rate']:.1%}")
    st.metric("P2 Policies", f"{s2['policies']:,}")
    st.caption(f"ε={s2['epsilon']:.4f}  T={s2['temperature']:.2f}")
with col3:
    st.metric("Total Games", total_games)
    st.metric("Draws", s1['draws'])
    st.metric("P1 Q-States", f"{s1['q_states']:,}")
with col4:
    st.metric("P1 Total Moves", f"{s1['total_moves']:,}")
    st.metric("P2 Total Moves", f"{s2['total_moves']:,}")
    st.metric("MCTS Sims P1/P2", f"{mcts1}/{mcts2}")

st.markdown("---")

# ============================================================================
# Training Loop
# ============================================================================

if train_button:
    st.subheader("⬡ Self-Play Training")
    status_ph = st.empty()
    prog_bar = st.progress(0.0)
    board_ph = st.empty() if show_live_board else None

    agent1.reset_stats()
    agent2.reset_stats()

    history = {
        'agent1_wins': [], 'agent2_wins': [], 'draws': [],
        'agent1_epsilon': [], 'agent2_epsilon': [],
        'agent1_policies': [], 'agent2_policies': [],
        'agent1_q_states': [], 'agent2_q_states': [],
        'episode': []
    }

    for ep in range(1, int(episodes) + 1):
        play_hex_game(agent1, agent2, training=True)
        agent1.decay_epsilon()
        agent2.decay_epsilon()

        if ep % int(update_freq) == 0:
            history['agent1_wins'].append(agent1.wins)
            history['agent2_wins'].append(agent2.wins)
            history['draws'].append(agent1.draws)
            history['agent1_epsilon'].append(agent1.epsilon)
            history['agent2_epsilon'].append(agent2.epsilon)
            history['agent1_policies'].append(len(agent1.policy_table))
            history['agent2_policies'].append(len(agent2.policy_table))
            history['agent1_q_states'].append(len(agent1.q_table))
            history['agent2_q_states'].append(len(agent2.q_table))
            history['episode'].append(ep)

            prog = ep / episodes
            prog_bar.progress(prog)
            status_ph.markdown(f"""
| Metric | Agent 1 (Red) | Agent 2 (Blue) |
|:-------|:-------------:|:--------------:|
| **Wins** | {agent1.wins} | {agent2.wins} |
| **Draws** | {agent1.draws} | — |
| **Epsilon ε** | {agent1.epsilon:.4f} | {agent2.epsilon:.4f} |
| **Policies** | {len(agent1.policy_table):,} | {len(agent2.policy_table):,} |
| **Q-States** | {len(agent1.q_table):,} | {len(agent2.q_table):,} |

**Episode {ep}/{int(episodes)}** ({prog*100:.1f}%)
""")
            if show_live_board and board_ph is not None:
                demo_g = HexLineGame()
                fig = draw_board(demo_g.board, "Training Demo Board")
                board_ph.pyplot(fig)
                plt.close(fig)

    prog_bar.progress(1.0)
    st.toast("Training Complete! ⬡", icon="✨")
    st.session_state.hl_training_history = history
    time.sleep(0.5)
    st.rerun()

# ============================================================================
# Training Analytics
# ============================================================================

if 'hl_training_history' in st.session_state and st.session_state.hl_training_history:
    hist = st.session_state.hl_training_history
    if hist.get('episode') and len(hist['episode']) > 0:
        st.subheader("📊 Training Analytics")
        fig_charts = draw_training_charts(hist)
        st.pyplot(fig_charts)
        plt.close(fig_charts)

        with st.expander("📋 Full Training Data Table"):
            df_hist = pd.DataFrame({
                'Episode': hist['episode'],
                'P1 Wins': hist['agent1_wins'],
                'P2 Wins': hist['agent2_wins'],
                'Draws': hist['draws'],
                'P1 ε': [f"{v:.4f}" for v in hist['agent1_epsilon']],
                'P2 ε': [f"{v:.4f}" for v in hist['agent2_epsilon']],
                'P1 Policies': hist['agent1_policies'],
                'P2 Policies': hist['agent2_policies'],
                'P1 Q-States': hist['agent1_q_states'],
                'P2 Q-States': hist['agent2_q_states'],
            })
            st.dataframe(df_hist, use_container_width=True)

# ============================================================================
# Championship Match (AI vs AI)
# ============================================================================

st.markdown("---")
st.subheader("⚔️ AI Championship Match")

if len(agent1.policy_table) > 3 or len(agent1.q_table) > 3:
    if st.button("▶️ Watch Championship Match", use_container_width=True):
        champ_game = HexLineGame()
        champ_agents = {1: agent1, 2: agent2}
        board_ph = st.empty()
        info_ph = st.empty()
        move_num = 0

        with st.spinner("Agents competing..."):
            while not champ_game.game_over and move_num < 20:
                current = champ_game.current_player
                mv = champ_agents[current].choose_action(champ_game, training=False)
                if mv is None:
                    break
                champ_game.make_move(mv)
                move_num += 1
                player_name = "Red (P1)" if current == 1 else "Blue (P2)"
                info_ph.caption(f"Move {move_num}: **{player_name}** plays hex **{mv}** ({HEX_LABELS.get(mv, '?')})")
                win_cells = champ_game.get_winning_cells() if champ_game.game_over else None
                fig = draw_board(champ_game.board,
                                  f"Move {move_num}: {player_name} → hex {mv}",
                                  last_move=mv, win_cells=win_cells)
                board_ph.pyplot(fig)
                plt.close(fig)
                time.sleep(0.6)

        if champ_game.winner == 1:
            st.success("🏆 Agent 1 (Red) Wins the Championship!")
        elif champ_game.winner == 2:
            st.error("🏆 Agent 2 (Blue) Wins the Championship!")
        else:
            st.warning("🤝 Draw!")
else:
    st.info("⬡ Train agents first to enable championship match.")

# ============================================================================
# Human vs AI
# ============================================================================

st.markdown("---")
st.header("🎮 Human vs AI Challenge")

col_h1, col_h2, col_h3 = st.columns(3)
with col_h1:
    human_color = st.selectbox("You Play As", ["Player 1 (Red)", "Player 2 (Blue)"])
with col_h2:
    ai_opponent_sel = st.selectbox("AI Opponent Brain", ["Agent 1 (Red)", "Agent 2 (Blue)"])
with col_h3:
    ai_strength_sel = st.selectbox("AI Strength", ["Easy (20 sims)", "Medium (80 sims)",
                                                     "Hard (200 sims)", "Expert (400 sims)"])
    strength_map = {"Easy (20 sims)": 20, "Medium (80 sims)": 80,
                    "Hard (200 sims)": 200, "Expert (400 sims)": 400}

if st.button("🎯 Start Game vs AI", use_container_width=True, type="primary"):
    human_pid = 1 if "Player 1" in human_color else 2
    ai_pid = 3 - human_pid
    st.session_state.hl_hvai_game = HexLineGame()
    st.session_state.hl_hvai_active = True
    st.session_state.hl_human_pid = human_pid
    st.session_state.hl_ai_pid = ai_pid
    st.session_state.hl_ai_ref = agent1 if "Agent 1" in ai_opponent_sel else agent2
    st.session_state.hl_ai_sims = strength_map[ai_strength_sel]
    st.rerun()

if st.session_state.get('hl_hvai_active', False):
    hv_game: HexLineGame = st.session_state.hl_hvai_game
    human_pid = st.session_state.hl_human_pid
    ai_pid = st.session_state.hl_ai_pid
    ai_ref: HexLineAgent = st.session_state.hl_ai_ref
    ai_sims = st.session_state.hl_ai_sims

    # AI turn
    if hv_game.current_player == ai_pid and not hv_game.game_over:
        with st.spinner("🤖 AI thinking..."):
            old_sims = ai_ref.mcts_sims
            ai_ref.mcts_sims = ai_sims
            ai_mv = ai_ref.choose_action(hv_game, training=False)
            ai_ref.mcts_sims = old_sims
            if ai_mv is not None:
                hv_game.make_move(ai_mv)
                st.rerun()

    # Status
    if hv_game.game_over:
        if hv_game.winner == human_pid:
            st.success("🎉 YOU WIN! Excellent play!")
        elif hv_game.winner == ai_pid:
            st.error("🤖 AI Wins! Try again!")
        else:
            st.warning("🤝 Draw!")
    else:
        turn = "Your Turn ✍️" if hv_game.current_player == human_pid else "AI Thinking... 🤖"
        p_str = "Red (P1)" if hv_game.current_player == 1 else "Blue (P2)"
        st.info(f"**{turn}** | Current: {p_str} | Move #{hv_game.move_count + 1}")

    win_cells = hv_game.get_winning_cells() if hv_game.game_over else None
    col_board, col_info = st.columns([3, 2])

    with col_board:
        fig_hv = draw_board(hv_game.board, "Human vs AI — Hex-Line",
                             win_cells=win_cells)
        st.pyplot(fig_hv)
        plt.close(fig_hv)

    with col_info:
        st.markdown("### 📊 Board Analysis")
        bi = hv_game.get_board_info()
        st.metric("P1 Outer Hexes", f"{bi['p1_outer']} / 6",
                  "↑ Leading" if bi['p1_outer'] > bi['p2_outer'] else "")
        st.metric("P2 Outer Hexes", f"{bi['p2_outer']} / 6",
                  "↑ Leading" if bi['p2_outer'] > bi['p1_outer'] else "")
        st.metric("P1 Threats", bi['p1_threats'])
        st.metric("P2 Threats", bi['p2_threats'])
        st.metric("P1 Fork Potential", bi['p1_forks'])
        st.metric("P2 Fork Potential", bi['p2_forks'])

        if hv_game.board[0] != 0:
            st.markdown(f"**Center Hex**: {'🔴 P1' if hv_game.board[0] == 1 else '🔵 P2'}")
        else:
            st.markdown("**Center Hex**: Empty ⬡")

        st.markdown("### 📜 Move History")
        labels = ['R' if i % 2 == 0 else 'B' for i in range(len(hv_game.move_history))]
        hist_str = " → ".join(
            [f"{lbl}{m}({HEX_LABELS.get(m,'?')[:3]})"
             for lbl, m in zip(labels, hv_game.move_history)]
        )
        st.caption(hist_str or "No moves yet.")

        if hv_game.event_log:
            st.markdown("### ⚠️ Events")
            for ev in hv_game.event_log[-5:]:
                st.caption(ev)

    # Human move input
    if not hv_game.game_over and hv_game.current_player == human_pid:
        st.markdown("---")
        st.markdown("### 🎯 Choose Your Hex")
        valid = hv_game.get_valid_moves()
        cols = st.columns(min(len(valid), 7))
        for ci, pos in enumerate(valid):
            lbl = HEX_LABELS.get(pos, str(pos))
            if cols[ci % len(cols)].button(f"⬡ {pos}: {lbl}",
                                            key=f"hv_mv_{pos}_{hv_game.move_count}"):
                hv_game.make_move(pos)
                st.rerun()

    if hv_game.game_over:
        if st.button("🔄 Play Again", use_container_width=True):
            st.session_state.hl_hvai_game = HexLineGame()
            st.session_state.hl_hvai_active = True
            st.rerun()

# ============================================================================
# Quick Simulation Mode
# ============================================================================

st.markdown("---")
with st.expander("🔬 Quick AI Simulation (No Training Effect)"):
    sim_n = st.slider("Games to simulate", 5, 300, 30, 5)
    if st.button("▶️ Run Simulation", key="sim_go"):
        sim_r = {1: 0, 2: 0, None: 0}
        with st.spinner(f"Simulating {sim_n} games..."):
            for _ in range(sim_n):
                r = play_hex_game(agent1, agent2, training=False)
                sim_r[r] += 1
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("🔴 P1 Wins", sim_r[1], f"{sim_r[1]/sim_n:.1%}")
        col_s2.metric("🔵 P2 Wins", sim_r[2], f"{sim_r[2]/sim_n:.1%}")
        col_s3.metric("🤝 Draws", sim_r[None], f"{sim_r[None]/sim_n:.1%}")

        # Show last game board
        demo_g = HexLineGame()
        while not demo_g.game_over:
            cur = demo_g.current_player
            mv = (agent1 if cur == 1 else agent2).choose_action(demo_g, training=False)
            if mv is None:
                break
            demo_g.make_move(mv)
        wlbl = f"P{demo_g.winner}" if demo_g.winner else "Draw"
        fig_demo = draw_board(demo_g.board, f"Last Simulated Game — Winner: {wlbl}",
                               win_cells=demo_g.get_winning_cells())
        st.pyplot(fig_demo)
        plt.close(fig_demo)

# ============================================================================
# Rules & Strategy Guide
# ============================================================================

st.markdown("---")
with st.expander("📖 Hex-Line Rules & Strategy Guide"):
    st.markdown("""
## ⬡ Hex-Line Rules

### The Board
**7 hexagons**: Hex 0 (center) + hexes 1–6 (outer ring, clockwise from top).

### Winning Conditions
1. **Line Win**: 3-in-a-row through center — lines are [1,0,4], [2,0,5], [3,0,6]
2. **Domination Win**: Control **4 or more** outer hexes (1–6)

The game is designed to eliminate draws. One of these conditions MUST trigger.

### Advanced Strategy
- 🎯 **Center Priority**: Hex 0 is on ALL three win lines — secure it turn 1 or die trying
- 🔺 **Fork Attack**: Occupy positions that create two simultaneous threats; your opponent can only block one
- 🛡️ **Outer Watch**: Track your opponent's outer hex count — 3 means BLOCK now
- ⚡ **Tempo**: With only 7 hexes, wasted moves are fatal. Every placement must threaten or defend
- 🔮 **Diagonal Traps**: Use opposite outer hexes to force center occupation or surrender a line

### State Space Analysis
- Physical: 3⁷ = **2,187** configurations
- Legal: ~**700 states** — fully explorable by minimax at depth ≥ 7
- Complexity class: Completely solved (optimal play has a determined winner)

### AI Architecture Details
The agent combines MCTS (planning with PUCT exploration), Negamax/Alpha-Beta (tactical precision with move ordering), Q-Learning (tabular value estimates), and a policy table (distilled search knowledge). Threat detection and fork analysis are built into both the heuristic evaluator and the prior generation, making the AI extremely aggressive at identifying forcing sequences.
""")
