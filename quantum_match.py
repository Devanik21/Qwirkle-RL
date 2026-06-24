import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import random
import json
import zipfile
import io
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import pandas as pd

# ============================================================================
# Page Config
# ============================================================================
st.set_page_config(
    page_title="Quantum Match Arena",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚛️"
)

st.title("⚛️ Quantum Match Arena")
st.markdown("""
A hidden-information deduction game on a **2×2 grid** with double-sided flip tokens.
Master the quantum of uncertainty — every flip changes the battlefield.

**AI Architecture:**
- 🌳 **MCTS + PUCT** — Monte Carlo Tree Search with AlphaZero's UCB formula
- 🧠 **Negamax + Alpha-Beta** — Full adversarial search with move ordering
- 🎯 **Quantum State Evaluator** — Heuristic policy mimicking a neural value head
- 🔄 **Self-Play Reinforcement** — Policy bootstrapping via experience tables
- 📊 **Q-Learning** — Tabular state-action value estimates
- 🔬 **Flip Sequence Analysis** — Threat detection across flip and place actions
""")

st.markdown("""
<style>
body { background-color: #0e1117; }
.stApp { background-color: #0e1117; }
.stButton>button {
    background: linear-gradient(90deg, #0d1b2a, #1b2838);
    color: #e0e0ff; border: 1px solid #334; border-radius: 8px; transition: all 0.2s;
}
.stButton>button:hover { border-color: #8888FF; color: #ccccff; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Quantum Match Game Logic
# ============================================================================

# Action types
PLACE = 'place'    # place token from pool onto empty square
FLIP  = 'flip'     # flip an existing token to own color

@dataclass
class QAction:
    action_type: str    # PLACE or FLIP
    position: int       # 0-3 (2x2 grid, row-major: 0=TL, 1=TR, 2=BL, 3=BR)
    player: int

    def __hash__(self):
        return hash((self.action_type, self.position, self.player))

    def __eq__(self, other):
        return (self.action_type == other.action_type and
                self.position == other.position and self.player == other.player)

    def to_key(self) -> str:
        return f"{self.action_type[0]}{self.position}{self.player}"

    @staticmethod
    def from_key(key: str, player: int) -> 'QAction':
        atype = PLACE if key[0] == 'p' else FLIP
        pos = int(key[1])
        return QAction(atype, pos, player)


class QuantumMatchGame:
    """
    Quantum Match: 2×2 grid, shared pool of 4 double-sided tokens.
    - PLACE: put a token from pool onto empty square, color-up facing player
    - FLIP: flip opponent's token to your color
    - Win: control all 4 squares at end of any turn
    - State space: 3^4 = 81 configurations (0=empty, 1=P1, 2=P2)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        # Board: 0=empty, 1=P1, 2=P2 (2x2 flattened: indices 0,1,2,3)
        self.board = [0, 0, 0, 0]
        # Token pool: how many tokens remain to be placed
        self.pool = 4
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history: List[QAction] = []
        self.move_count = 0
        self.event_log: List[str] = []
        return self.get_state()

    def get_state(self) -> tuple:
        return tuple(self.board) + (self.current_player, self.pool)

    def get_state_key(self) -> str:
        return ''.join(map(str, self.board)) + str(self.current_player) + str(self.pool)

    def copy(self) -> 'QuantumMatchGame':
        g = QuantumMatchGame()
        g.board = self.board[:]
        g.pool = self.pool
        g.current_player = self.current_player
        g.game_over = self.game_over
        g.winner = self.winner
        g.move_history = self.move_history[:]
        g.move_count = self.move_count
        g.event_log = self.event_log[:]
        return g

    def get_valid_actions(self) -> List[QAction]:
        if self.game_over:
            return []
        actions = []
        p = self.current_player
        opp_color = 3 - p

        # PLACE: from pool onto empty squares
        if self.pool > 0:
            for pos in range(4):
                if self.board[pos] == 0:
                    actions.append(QAction(PLACE, pos, p))

        # FLIP: flip opponent's token
        for pos in range(4):
            if self.board[pos] == opp_color:
                actions.append(QAction(FLIP, pos, p))

        return actions

    def make_action(self, action: QAction) -> Tuple[tuple, float, bool]:
        if self.game_over:
            return self.get_state(), 0.0, True

        reward = 0.0
        p = self.current_player

        if action.action_type == PLACE:
            if self.board[action.position] != 0 or self.pool <= 0:
                return self.get_state(), -1.0, False
            self.board[action.position] = p
            self.pool -= 1
            reward = 0.5

        elif action.action_type == FLIP:
            if self.board[action.position] != (3 - p):
                return self.get_state(), -1.0, False
            self.board[action.position] = p
            reward = 1.0  # Flipping is aggressive and valuable

        self.move_history.append(action)
        self.move_count += 1

        # Win check: all 4 squares = current player
        if all(c == p for c in self.board):
            self.game_over = True
            self.winner = p
            reward = 100.0
            self.event_log.append(f"P{p} wins by controlling all 4 squares!")
        else:
            self.current_player = 3 - p

        return self.get_state(), reward, self.game_over

    def check_win(self, player: int) -> bool:
        return all(c == player for c in self.board)

    def count_owned(self, player: int) -> int:
        return sum(1 for c in self.board if c == player)

    def evaluate_position(self, player: int) -> float:
        """
        Rich evaluation: ownership, flip threats, board control dynamics.
        """
        if self.winner == player:
            return 100000.0
        if self.winner is not None and self.winner != player:
            return -100000.0

        opponent = 3 - player
        score = 0.0

        my_cnt = self.count_owned(player)
        op_cnt = self.count_owned(opponent)
        empty = sum(1 for c in self.board if c == 0)

        # Ownership advantage
        score += (my_cnt - op_cnt) * 200

        # One-flip-from-win: extremely dangerous
        if my_cnt == 3 and op_cnt == 1:
            score += 800  # Can flip opponent's last one
        if op_cnt == 3 and my_cnt == 1:
            score -= 800  # Opponent can flip my last one

        # Pool awareness: fewer pool tokens = more dynamic flipping game
        if self.pool == 0:
            # Pure flip game — advantage amplified by imbalance
            score += (my_cnt - op_cnt) * 400
        else:
            # Placement phase: empty squares add uncertainty, prefer claiming them
            score += empty * 15 * (1 if my_cnt >= op_cnt else -1)

        # Flip threat analysis
        my_flippable = sum(1 for pos in range(4) if self.board[pos] == opponent)
        op_flippable = sum(1 for pos in range(4) if self.board[pos] == player)

        # Having more to flip = more power
        score += my_flippable * 50
        score -= op_flippable * 50

        # Consecutive control bonus: center positions (1,2) vs corners (0,3)
        # In 2x2 grid all are equivalent structurally, but diagonal pairs matter
        my_diag1 = (self.board[0] == player and self.board[3] == player)
        my_diag2 = (self.board[1] == player and self.board[2] == player)
        op_diag1 = (self.board[0] == opponent and self.board[3] == opponent)
        op_diag2 = (self.board[1] == opponent and self.board[2] == opponent)
        score += (my_diag1 + my_diag2) * 100
        score -= (op_diag1 + op_diag2) * 100

        # Mobility: count available actions
        orig_cp = self.current_player
        self.current_player = player
        my_moves = len(self.get_valid_actions())
        self.current_player = opponent
        op_moves = len(self.get_valid_actions())
        self.current_player = orig_cp
        score += (my_moves - op_moves) * 30

        # Board saturation bonus (when all squares filled, pure flip war)
        if empty == 0:
            score += (my_cnt - op_cnt) * 600

        return score

    def get_threat_info(self) -> Dict:
        """Return threat analysis for UI display."""
        info = {}
        for p in [1, 2]:
            owned = self.count_owned(p)
            opp = 3 - p
            flippable = sum(1 for c in self.board if c == opp)
            can_win_flip = (owned == 3 and flippable >= 1)
            can_win_place = (owned == 3 and self.pool > 0 and
                             any(c == 0 for c in self.board))
            info[f'p{p}_owned'] = owned
            info[f'p{p}_flippable'] = flippable
            info[f'p{p}_can_win'] = can_win_flip or can_win_place
            info[f'p{p}_threat_type'] = (
                'FLIP WIN!' if can_win_flip else
                'PLACE WIN!' if can_win_place else
                'None'
            )
        info['pool'] = self.pool
        info['empty'] = sum(1 for c in self.board if c == 0)
        return info

    def get_action_quality_labels(self) -> Dict[str, str]:
        """Label each legal action with a quality hint."""
        labels = {}
        p = self.current_player
        owned = self.count_owned(p)

        for action in self.get_valid_actions():
            key = action.to_key()
            if action.action_type == FLIP:
                # Check if flipping leads to win
                sim = self.copy()
                sim.make_action(action)
                if sim.winner == p:
                    labels[key] = "⚡ WIN!"
                elif owned >= 3:
                    labels[key] = "🔥 Strong"
                else:
                    labels[key] = "🔄 Flip"
            else:
                sim = self.copy()
                sim.make_action(action)
                if sim.winner == p:
                    labels[key] = "⚡ WIN!"
                else:
                    labels[key] = "📍 Place"
        return labels

# ============================================================================
# MCTS Node
# ============================================================================

class QMCTSNode:
    def __init__(self, game: QuantumMatchGame, parent=None,
                 action: Optional[QAction] = None, prior: float = 1.0):
        self.game = game
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children: Dict[str, 'QMCTSNode'] = {}
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

    def select_child(self, c_puct: float = 1.5) -> 'QMCTSNode':
        return max(self.children.values(),
                   key=lambda c: c.ucb_score(self.visit_count, c_puct))

    def expand(self, policy_priors: Dict[str, float]):
        actions = self.game.get_valid_actions()
        if not actions:
            return
        total = sum(policy_priors.values()) or len(actions)
        for act in actions:
            key = act.to_key()
            child_game = self.game.copy()
            child_game.make_action(act)
            prior = policy_priors.get(key, 1.0) / total
            self.children[key] = QMCTSNode(child_game, parent=self, action=act, prior=prior)
        self.is_expanded = True

    def backup(self, value: float):
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)

# ============================================================================
# AlphaZero-Inspired Quantum Match Agent
# ============================================================================

class QuantumAgent:
    """
    Full AlphaZero-inspired agent for Quantum Match.
    MCTS + Negamax + Q-Learning + Policy Table.
    Handles mixed PLACE/FLIP action spaces.
    """
    def __init__(self, player_id: int, lr: float = 0.3, gamma: float = 0.97,
                 epsilon: float = 1.0, mcts_sims: int = 200, minimax_depth: int = 8):
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

        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.policy_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.visit_table: Dict[str, int] = defaultdict(int)

        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_moves = 0
        self.flip_wins = 0    # Wins by flipping last tile
        self.place_wins = 0   # Wins by placing last tile

    def get_policy_priors(self, game: QuantumMatchGame) -> Dict[str, float]:
        state_key = game.get_state_key()
        actions = game.get_valid_actions()
        priors = {}

        for act in actions:
            key = act.to_key()
            learned = self.policy_table[state_key].get(key, 0.0)
            q_val = self.q_table[state_key].get(key, 0.0)
            prior = 1.0 + max(0, learned) + max(0, q_val) * 0.5

            # Immediate win check
            sim = game.copy()
            sim.make_action(act)
            if sim.winner == game.current_player:
                priors[key] = prior + 10000.0
                continue

            # Opponent block: check if opponent can win on next move
            opp = 3 - game.current_player
            opp_game = game.copy()
            opp_game.make_action(act)
            for opp_act in opp_game.get_valid_actions():
                sim2 = opp_game.copy()
                sim2.make_action(opp_act)
                if sim2.winner == opp:
                    prior += 400.0

            # Flip preference when board is nearly full
            if act.action_type == FLIP:
                my_cnt = game.count_owned(game.current_player)
                if my_cnt >= 2:
                    prior += 100.0 * (my_cnt / 3)
                prior += 50.0  # Flipping is generally powerful

            # Placement: prefer filling board early
            if act.action_type == PLACE and game.pool > 2:
                prior += 30.0

            priors[key] = max(0.01, prior)

        return priors

    def mcts_search(self, game: QuantumMatchGame) -> QMCTSNode:
        root = QMCTSNode(game.copy())
        for _ in range(self.mcts_sims):
            node = root
            sim_game = game.copy()
            while node.is_expanded and node.children and not sim_game.game_over:
                node = node.select_child(self.c_puct)
                sim_game.make_action(node.action)
            if not sim_game.game_over:
                priors = self.get_policy_priors(sim_game)
                node.expand(priors)
            value = self._evaluate_leaf(sim_game)
            node.backup(value)
        return root

    def _evaluate_leaf(self, game: QuantumMatchGame) -> float:
        if game.game_over:
            if game.winner == self.player_id:
                return 1.0
            elif game.winner is not None:
                return -1.0
            return 0.0
        score = self._negamax(game, self.minimax_depth, -float('inf'), float('inf'),
                               game.current_player == self.player_id)
        return math.tanh(score / 500.0)

    def _negamax(self, game: QuantumMatchGame, depth: int,
                 alpha: float, beta: float, maximizing: bool) -> float:
        if depth == 0 or game.game_over:
            return game.evaluate_position(self.player_id)
        actions = game.get_valid_actions()
        if not actions:
            return game.evaluate_position(self.player_id)

        # Move ordering: evaluate actions heuristically
        scored = []
        for act in actions:
            sim = game.copy()
            sim.make_action(act)
            scored.append((act, sim.evaluate_position(self.player_id)))
        scored.sort(key=lambda x: x[1], reverse=maximizing)

        if maximizing:
            best = -float('inf')
            for act, _ in scored:
                sim = game.copy()
                sim.make_action(act)
                val = self._negamax(sim, depth - 1, alpha, beta, False)
                best = max(best, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return best
        else:
            best = float('inf')
            for act, _ in scored:
                sim = game.copy()
                sim.make_action(act)
                val = self._negamax(sim, depth - 1, alpha, beta, True)
                best = min(best, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return best

    def choose_action(self, game: QuantumMatchGame,
                      training: bool = True) -> Optional[QAction]:
        actions = game.get_valid_actions()
        if not actions:
            return None

        # Immediate win
        for act in actions:
            sim = game.copy()
            sim.make_action(act)
            if sim.winner == self.player_id:
                self.total_moves += 1
                return act

        # Block opponent win — handled by negamax depth search
        if training and random.random() < self.epsilon:
            self.total_moves += 1
            return random.choice(actions)

        root = self.mcts_search(game)
        if not root.children:
            return random.choice(actions)

        if training and self.temperature > 0.1:
            visits = {key: c.visit_count for key, c in root.children.items()}
            total = sum(visits.values())
            if total > 0:
                keys = list(visits.keys())
                probs = [visits[k] / total for k in keys]
                chosen_key = random.choices(keys, weights=probs)[0]
                chosen = root.children[chosen_key].action
            else:
                chosen = random.choice(actions)
        else:
            best_key = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
            chosen = root.children[best_key].action

        # Update policy table
        state_key = game.get_state_key()
        total_v = sum(c.visit_count for c in root.children.values())
        for key, child in root.children.items():
            self.policy_table[state_key][key] = child.visit_count / max(1, total_v)

        self.total_moves += 1
        return chosen

    def update_from_game(self, history: List[Tuple[str, str, int]],
                         result: Optional[int]):
        for state_key, action_key, player in reversed(history):
            if player != self.player_id:
                continue
            if result == self.player_id:
                reward = 1.0
            elif result is None:
                reward = 0.0
            else:
                reward = -1.0
            old_q = self.q_table[state_key][action_key]
            self.q_table[state_key][action_key] = old_q + self.lr * (reward - old_q)
            old_p = self.policy_table[state_key][action_key]
            self.policy_table[state_key][action_key] = old_p + self.lr * (reward - old_p)
            self.visit_table[state_key] += 1

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.temperature = max(0.1, self.temperature * 0.99)

    def reset_stats(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_moves = 0
        self.flip_wins = 0
        self.place_wins = 0

    def get_stats(self) -> Dict:
        total = self.wins + self.losses + self.draws
        return {
            'wins': self.wins, 'losses': self.losses, 'draws': self.draws,
            'total': total, 'win_rate': self.wins / max(1, total),
            'policies': len(self.policy_table), 'q_states': len(self.q_table),
            'epsilon': self.epsilon, 'temperature': self.temperature,
            'total_moves': self.total_moves,
            'flip_wins': self.flip_wins, 'place_wins': self.place_wins,
        }

# ============================================================================
# Self-Play Training
# ============================================================================

def play_qm_game(agent1: QuantumAgent, agent2: QuantumAgent,
                 training: bool = True) -> Optional[int]:
    game = QuantumMatchGame()
    history: List[Tuple[str, str, int]] = []
    agents = {1: agent1, 2: agent2}
    max_moves = 80

    while not game.game_over and game.move_count < max_moves:
        current = game.current_player
        agent = agents[current]
        state_key = game.get_state_key()
        action = agent.choose_action(game, training)
        if action is None:
            break
        history.append((state_key, action.to_key(), current))
        _, _, done = game.make_action(action)

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

CELL_LABELS = {0: "Top-Left", 1: "Top-Right", 2: "Bot-Left", 3: "Bot-Right"}

def draw_qm_board(board: List[int], pool: int, title: str = "Quantum Match",
                   last_action: Optional[QAction] = None,
                   win_flash: bool = False) -> plt.Figure:
    """Draw the 2×2 Quantum Match grid."""
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    cell_colors = {0: '#1a1a2e', 1: '#1a0a0a', 2: '#0a0a1a'}
    token_colors = {0: None, 1: '#DC143C', 2: '#1E90FF'}
    edge_colors = {0: '#333355', 1: '#8B0000', 2: '#00008B'}

    # Position map: (col, row) for each cell index
    positions = {0: (0, 1), 1: (1, 1), 2: (0, 0), 3: (1, 0)}

    for idx in range(4):
        col, row = positions[idx]
        cell_val = board[idx]

        is_last = last_action and last_action.position == idx
        is_win = win_flash

        face = cell_colors[cell_val]
        edge = edge_colors[cell_val]
        lw = 1.5
        if is_last and last_action.action_type == FLIP:
            edge = '#FFD700'
            lw = 4
        elif is_last and last_action.action_type == PLACE:
            edge = '#FFFFFF'
            lw = 3
        if is_win and cell_val != 0:
            face = '#1a2a0a'
            edge = '#00FF44'
            lw = 4

        rect = plt.Rectangle((col * 2.2, row * 2.2), 2.0, 2.0,
                               facecolor=face, edgecolor=edge, linewidth=lw)
        ax.add_patch(rect)

        # Cell label
        ax.text(col * 2.2 + 1.0, row * 2.2 + 0.15, f'{idx}:{CELL_LABELS[idx][:3]}',
                ha='center', va='bottom', fontsize=8, color='#555577')

        # Token
        if cell_val != 0:
            color = token_colors[cell_val]
            circle = plt.Circle((col * 2.2 + 1.0, row * 2.2 + 1.15), 0.6,
                                  color=color, ec='#ffffff', linewidth=2, zorder=3)
            ax.add_patch(circle)
            ax.text(col * 2.2 + 1.0, row * 2.2 + 1.15, '●',
                    ha='center', va='center', fontsize=30,
                    color=token_colors[cell_val], zorder=4, fontweight='bold')

        # Action indicator
        if is_last and last_action.action_type == FLIP:
            ax.text(col * 2.2 + 1.0, row * 2.2 + 1.75, '🔄',
                    ha='center', va='center', fontsize=18, zorder=5)
        elif is_last and last_action.action_type == PLACE:
            ax.text(col * 2.2 + 1.0, row * 2.2 + 1.75, '📍',
                    ha='center', va='center', fontsize=18, zorder=5)

    # Grid lines
    for i in range(3):
        x = i * 2.2
        ax.axvline(x=x, color='#334455', linewidth=1, alpha=0.6)
        ax.axhline(y=x, color='#334455', linewidth=1, alpha=0.6)

    # Pool indicator
    pool_str = "⬟ " * pool + "◻ " * (4 - pool)
    ax.text(2.2, -0.35, f"Pool: {pool_str}({pool} left)",
            ha='center', va='center', fontsize=10, color='#AAAACC')

    ax.set_xlim(-0.1, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, color='#CCCCFF', fontweight='bold', pad=12)

    p1_patch = mpatches.Patch(color='#DC143C', label='Player 1 (Red)')
    p2_patch = mpatches.Patch(color='#1E90FF', label='Player 2 (Blue)')
    ax.legend(handles=[p1_patch, p2_patch], loc='lower right',
              facecolor='#0d1117', edgecolor='#334455', labelcolor='white', fontsize=9)

    return fig


def draw_action_history_chart(move_history: List[QAction]) -> plt.Figure:
    """Visualize action sequence: PLACE vs FLIP over turns."""
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#1a1a2e')

    for spine in ax.spines.values():
        spine.set_edgecolor('#334455')
    ax.tick_params(colors='#AAAACC')

    turns = list(range(1, len(move_history) + 1))
    flip_turns = [i + 1 for i, a in enumerate(move_history) if a.action_type == FLIP]
    place_turns = [i + 1 for i, a in enumerate(move_history) if a.action_type == PLACE]
    p1_turns = [t for t, a in zip(turns, move_history) if a.player == 1]
    p2_turns = [t for t, a in zip(turns, move_history) if a.player == 2]

    ax.scatter(flip_turns, [1.2] * len(flip_turns), marker='v',
               color='#FFD700', s=100, label='FLIP', zorder=3)
    ax.scatter(place_turns, [0.8] * len(place_turns), marker='^',
               color='#AAFFAA', s=100, label='PLACE', zorder=3)

    for t in p1_turns:
        ax.axvline(x=t, color='#DC143C', alpha=0.3, linewidth=2)
    for t in p2_turns:
        ax.axvline(x=t, color='#1E90FF', alpha=0.3, linewidth=2)

    ax.set_xlim(0, len(move_history) + 1)
    ax.set_ylim(0, 2)
    ax.set_yticks([0.8, 1.2])
    ax.set_yticklabels(['PLACE', 'FLIP'], color='#AAAACC')
    ax.set_xlabel('Turn', color='#AAAACC')
    ax.set_title('Action Sequence', color='#CCCCFF', fontweight='bold')
    ax.legend(facecolor='#1a1a2e', edgecolor='#334455', labelcolor='white')

    return fig


def draw_training_charts(history: Dict) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes.flat:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#AAAACC')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334455')

    eps = history.get('episode', [])
    if not eps:
        return fig

    axes[0, 0].plot(eps, history.get('agent1_wins', []), color='#DC143C', lw=2, label='P1')
    axes[0, 0].plot(eps, history.get('agent2_wins', []), color='#1E90FF', lw=2, label='P2')
    axes[0, 0].plot(eps, history.get('draws', []), color='#888888', lw=1.5, ls='--', label='Draws')
    axes[0, 0].set_title('Win/Draw Distribution', color='#CCCCFF')
    axes[0, 0].legend(facecolor='#1a1a2e', edgecolor='#334455', labelcolor='white')

    axes[0, 1].plot(eps, history.get('agent1_epsilon', []), color='#FF6B6B', lw=2, label='P1 ε')
    axes[0, 1].plot(eps, history.get('agent2_epsilon', []), color='#66B3FF', lw=2, label='P2 ε')
    axes[0, 1].set_title('Exploration Rate (ε)', color='#CCCCFF')
    axes[0, 1].legend(facecolor='#1a1a2e', edgecolor='#334455', labelcolor='white')

    axes[1, 0].plot(eps, history.get('agent1_policies', []), color='#FF6B6B', lw=2, label='P1')
    axes[1, 0].plot(eps, history.get('agent2_policies', []), color='#66B3FF', lw=2, label='P2')
    axes[1, 0].set_title('Policy Table Size', color='#CCCCFF')
    axes[1, 0].legend(facecolor='#1a1a2e', edgecolor='#334455', labelcolor='white')

    a1w = history.get('agent1_wins', [0])
    a2w = history.get('agent2_wins', [0])
    dr = history.get('draws', [0])
    totals = [max(1, a + b + d) for a, b, d in zip(a1w, a2w, dr)]
    axes[1, 1].plot(eps, [w / t for w, t in zip(a1w, totals)], color='#DC143C', lw=2, label='P1 WR')
    axes[1, 1].plot(eps, [w / t for w, t in zip(a2w, totals)], color='#1E90FF', lw=2, label='P2 WR')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Win Rate Over Time', color='#CCCCFF')
    axes[1, 1].legend(facecolor='#1a1a2e', edgecolor='#334455', labelcolor='white')

    fig.suptitle('⚛️ Quantum Match Training Analytics', fontsize=15,
                 color='#CCCCFF', fontweight='bold')
    plt.tight_layout()
    return fig

# ============================================================================
# Save / Load
# ============================================================================

def serialize_agent(agent: QuantumAgent, role: str) -> Dict:
    return {
        'metadata': {'role': role, 'version': '1.0', 'game': 'quantum_match'},
        'player_id': agent.player_id,
        'epsilon': float(agent.epsilon),
        'temperature': float(agent.temperature),
        'wins': int(agent.wins), 'losses': int(agent.losses), 'draws': int(agent.draws),
        'total_moves': int(agent.total_moves),
        'flip_wins': int(agent.flip_wins), 'place_wins': int(agent.place_wins),
        'mcts_sims': int(agent.mcts_sims),
        'minimax_depth': int(agent.minimax_depth),
        'q_table': {sk: {ak: float(v) for ak, v in avs.items()}
                    for sk, avs in agent.q_table.items()},
        'policy_table': {sk: {ak: float(v) for ak, v in avs.items()}
                         for sk, avs in agent.policy_table.items()},
    }

def deserialize_agent(data: Dict, player_id: int) -> QuantumAgent:
    agent = QuantumAgent(player_id=player_id,
                         mcts_sims=data.get('mcts_sims', 200),
                         minimax_depth=data.get('minimax_depth', 8))
    agent.epsilon = data.get('epsilon', 0.1)
    agent.temperature = data.get('temperature', 0.3)
    agent.wins = data.get('wins', 0)
    agent.losses = data.get('losses', 0)
    agent.draws = data.get('draws', 0)
    agent.total_moves = data.get('total_moves', 0)
    agent.flip_wins = data.get('flip_wins', 0)
    agent.place_wins = data.get('place_wins', 0)
    for sk, avs in data.get('q_table', {}).items():
        for ak, v in avs.items():
            agent.q_table[sk][ak] = float(v)
    for sk, avs in data.get('policy_table', {}).items():
        for ak, v in avs.items():
            agent.policy_table[sk][ak] = float(v)
    return agent

def create_agents_zip(agent1: QuantumAgent, agent2: QuantumAgent, config: Dict) -> io.BytesIO:
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
                st.error("❌ Corrupt file.")
                return None, None, None
            d1 = json.loads(zf.read('agent1.json'))
            d2 = json.loads(zf.read('agent2.json'))
            cfg = json.loads(zf.read('config.json'))
        return deserialize_agent(d1, 1), deserialize_agent(d2, 2), cfg
    except Exception as e:
        st.error(f"❌ Load error: {e}")
        return None, None, None

# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.header("⚙️ Quantum Match Controls")

with st.sidebar.expander("1. Agent 1 (Red) Parameters", expanded=True):
    lr1 = st.slider("Learning Rate α₁", 0.05, 1.0, 0.25, 0.05)
    gamma1 = st.slider("Discount γ₁", 0.80, 0.99, 0.97, 0.01)
    mcts1 = st.slider("MCTS Simulations₁", 10, 800, 200, 10)
    mm1 = st.slider("Minimax Depth₁", 1, 12, 8, 1)

with st.sidebar.expander("2. Agent 2 (Blue) Parameters", expanded=True):
    lr2 = st.slider("Learning Rate α₂", 0.05, 1.0, 0.25, 0.05)
    gamma2 = st.slider("Discount γ₂", 0.80, 0.99, 0.97, 0.01)
    mcts2 = st.slider("MCTS Simulations₂", 10, 800, 150, 10)
    mm2 = st.slider("Minimax Depth₂", 1, 12, 7, 1)

with st.sidebar.expander("3. Training Configuration", expanded=True):
    episodes = st.number_input("Training Episodes", 10, 50000, 500, 50)
    update_freq = st.number_input("Update Every N Games", 1, 500, 25, 5)
    show_live = st.checkbox("Show Live Board During Training", False)

with st.sidebar.expander("4. Brain Storage", expanded=False):
    if 'qm_agent1' in st.session_state and st.session_state.qm_agent1:
        a1r = st.session_state.qm_agent1
        a2r = st.session_state.qm_agent2
        st.markdown("### 🧠 Neural Sync")
        c1, c2 = st.columns(2)
        if c1.button("P1 ➡️ P2"):
            st.session_state.qm_agent2.policy_table = deepcopy(a1r.policy_table)
            st.session_state.qm_agent2.q_table = deepcopy(a1r.q_table)
            st.session_state.qm_agent2.epsilon = a1r.epsilon
            st.toast("P2 now has P1's brain!", icon="🔵")
        if c2.button("P2 ➡️ P1"):
            st.session_state.qm_agent1.policy_table = deepcopy(a2r.policy_table)
            st.session_state.qm_agent1.q_table = deepcopy(a2r.q_table)
            st.session_state.qm_agent1.epsilon = a2r.epsilon
            st.toast("P1 now has P2's brain!", icon="🔴")
        st.markdown("---")
        cfg_save = {'lr1': lr1, 'gamma1': gamma1, 'mcts1': mcts1, 'mm1': mm1,
                    'lr2': lr2, 'gamma2': gamma2, 'mcts2': mcts2, 'mm2': mm2}
        zip_b = create_agents_zip(a1r, a2r, cfg_save)
        st.download_button("💾 Download Agents", zip_b,
                           "qmatch_agents.zip", "application/zip",
                           use_container_width=True)
    else:
        st.info("Train agents first to enable save.")
    st.markdown("---")
    upf = st.file_uploader("📤 Upload Agents (.zip)", type="zip")
    if upf and st.button("🔄 Load Agents", use_container_width=True):
        a1l, a2l, cfgl = load_agents_from_zip(upf)
        if a1l and a2l:
            st.session_state.qm_agent1 = a1l
            st.session_state.qm_agent2 = a2l
            st.toast("✅ Agents loaded!", icon="🧠")
            st.rerun()

train_btn = st.sidebar.button("⚛️ Begin Self-Play Training",
                               use_container_width=True, type="primary")
if st.sidebar.button("🧹 Reset Arena", use_container_width=True):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# ============================================================================
# Init Agents
# ============================================================================

if 'qm_agent1' not in st.session_state:
    st.session_state.qm_agent1 = QuantumAgent(1, lr1, gamma1, mcts_sims=mcts1, minimax_depth=mm1)
    st.session_state.qm_agent2 = QuantumAgent(2, lr2, gamma2, mcts_sims=mcts2, minimax_depth=mm2)

agent1: QuantumAgent = st.session_state.qm_agent1
agent2: QuantumAgent = st.session_state.qm_agent2

agent1.mcts_sims = mcts1; agent1.minimax_depth = mm1; agent1.lr = lr1
agent2.mcts_sims = mcts2; agent2.minimax_depth = mm2; agent2.lr = lr2

# ============================================================================
# Stats Dashboard
# ============================================================================

st.markdown("---")
s1 = agent1.get_stats()
s2 = agent2.get_stats()
total_g = s1['wins'] + s2['wins'] + s1['draws']

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
    st.metric("Total Games", total_g)
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

if train_btn:
    st.subheader("⚛️ Quantum Match Self-Play Training")
    status_ph = st.empty()
    prog_bar = st.progress(0.0)
    board_ph = st.empty() if show_live else None

    agent1.reset_stats()
    agent2.reset_stats()

    hist = {
        'agent1_wins': [], 'agent2_wins': [], 'draws': [],
        'agent1_epsilon': [], 'agent2_epsilon': [],
        'agent1_policies': [], 'agent2_policies': [],
        'agent1_q_states': [], 'agent2_q_states': [],
        'episode': []
    }

    for ep in range(1, int(episodes) + 1):
        play_qm_game(agent1, agent2, training=True)
        agent1.decay_epsilon()
        agent2.decay_epsilon()

        if ep % int(update_freq) == 0:
            hist['agent1_wins'].append(agent1.wins)
            hist['agent2_wins'].append(agent2.wins)
            hist['draws'].append(agent1.draws)
            hist['agent1_epsilon'].append(agent1.epsilon)
            hist['agent2_epsilon'].append(agent2.epsilon)
            hist['agent1_policies'].append(len(agent1.policy_table))
            hist['agent2_policies'].append(len(agent2.policy_table))
            hist['agent1_q_states'].append(len(agent1.q_table))
            hist['agent2_q_states'].append(len(agent2.q_table))
            hist['episode'].append(ep)

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
            if show_live and board_ph is not None:
                demo_g = QuantumMatchGame()
                fig = draw_qm_board(demo_g.board, demo_g.pool, "Training Demo")
                board_ph.pyplot(fig)
                plt.close(fig)

    prog_bar.progress(1.0)
    st.toast("Training Complete! ⚛️", icon="✨")
    st.session_state.qm_training_history = hist
    time.sleep(0.5)
    st.rerun()

# ============================================================================
# Training Analytics
# ============================================================================

if 'qm_training_history' in st.session_state and st.session_state.qm_training_history:
    th = st.session_state.qm_training_history
    if th.get('episode') and len(th['episode']) > 0:
        st.subheader("📊 Training Analytics")
        fig_c = draw_training_charts(th)
        st.pyplot(fig_c)
        plt.close(fig_c)

        with st.expander("📋 Full Training Data Table"):
            df_th = pd.DataFrame({
                'Episode': th['episode'],
                'P1 Wins': th['agent1_wins'], 'P2 Wins': th['agent2_wins'],
                'Draws': th['draws'],
                'P1 ε': [f"{v:.4f}" for v in th['agent1_epsilon']],
                'P2 ε': [f"{v:.4f}" for v in th['agent2_epsilon']],
                'P1 Policies': th['agent1_policies'], 'P2 Policies': th['agent2_policies'],
            })
            st.dataframe(df_th, use_container_width=True)

# ============================================================================
# Championship Match
# ============================================================================

st.markdown("---")
st.subheader("⚔️ AI Championship Match")

if len(agent1.policy_table) > 3 or len(agent1.q_table) > 3:
    if st.button("▶️ Watch Championship Match", use_container_width=True):
        champ = QuantumMatchGame()
        champ_agents = {1: agent1, 2: agent2}
        board_ph = st.empty()
        info_ph = st.empty()
        hist_ph = st.empty()
        mn = 0

        with st.spinner("Agents competing..."):
            while not champ.game_over and mn < 50:
                cur = champ.current_player
                act = champ_agents[cur].choose_action(champ, training=False)
                if act is None:
                    break
                champ.make_action(act)
                mn += 1
                pname = "Red (P1)" if cur == 1 else "Blue (P2)"
                aname = f"PLACE→cell {act.position}" if act.action_type == PLACE else f"FLIP cell {act.position}"
                info_ph.caption(f"Move {mn}: **{pname}** does **{aname}**")

                board_ph_fig = draw_qm_board(
                    champ.board, champ.pool,
                    f"Move {mn}: {pname} — {aname}",
                    last_action=act,
                    win_flash=champ.game_over
                )
                board_ph.pyplot(board_ph_fig)
                plt.close(board_ph_fig)

                # Action history
                if len(champ.move_history) > 0:
                    hist_fig = draw_action_history_chart(champ.move_history)
                    hist_ph.pyplot(hist_fig)
                    plt.close(hist_fig)

                time.sleep(0.7)

        if champ.winner == 1:
            st.success("🏆 Agent 1 (Red) controls ALL 4 squares — Wins!")
        elif champ.winner == 2:
            st.error("🏆 Agent 2 (Blue) controls ALL 4 squares — Wins!")
        else:
            st.warning("🤝 Draw or max moves reached.")
else:
    st.info("⚛️ Train agents first to enable championship match.")

# ============================================================================
# Human vs AI
# ============================================================================

st.markdown("---")
st.header("🎮 Human vs AI Challenge")

col_h1, col_h2, col_h3 = st.columns(3)
with col_h1:
    human_col = st.selectbox("You Play As", ["Player 1 (Red)", "Player 2 (Blue)"])
with col_h2:
    ai_opp = st.selectbox("AI Opponent Brain", ["Agent 1 (Red)", "Agent 2 (Blue)"])
with col_h3:
    ai_str = st.selectbox("AI Strength", ["Easy (20)", "Medium (80)", "Hard (200)", "Expert (500)"])
    s_map = {"Easy (20)": 20, "Medium (80)": 80, "Hard (200)": 200, "Expert (500)": 500}

if st.button("🎯 Start Human vs AI Game", use_container_width=True, type="primary"):
    human_pid = 1 if "Player 1" in human_col else 2
    ai_pid = 3 - human_pid
    st.session_state.qm_hv_game = QuantumMatchGame()
    st.session_state.qm_hv_active = True
    st.session_state.qm_human_pid = human_pid
    st.session_state.qm_ai_pid = ai_pid
    st.session_state.qm_ai_ref = agent1 if "Agent 1" in ai_opp else agent2
    st.session_state.qm_ai_sims = s_map[ai_str]
    st.session_state.qm_last_action = None
    st.rerun()

if st.session_state.get('qm_hv_active', False):
    hv_g: QuantumMatchGame = st.session_state.qm_hv_game
    h_pid = st.session_state.qm_human_pid
    ai_pid = st.session_state.qm_ai_pid
    ai_r: QuantumAgent = st.session_state.qm_ai_ref
    ai_sims = st.session_state.qm_ai_sims
    last_act = st.session_state.get('qm_last_action', None)

    # AI turn
    if hv_g.current_player == ai_pid and not hv_g.game_over:
        with st.spinner("🤖 AI thinking..."):
            old = ai_r.mcts_sims
            ai_r.mcts_sims = ai_sims
            ai_mv = ai_r.choose_action(hv_g, training=False)
            ai_r.mcts_sims = old
            if ai_mv:
                hv_g.make_action(ai_mv)
                st.session_state.qm_last_action = ai_mv
                st.rerun()

    # Status
    if hv_g.game_over:
        if hv_g.winner == h_pid:
            st.success("🎉 YOU WIN! You controlled all 4 squares!")
        elif hv_g.winner == ai_pid:
            st.error("🤖 AI Wins — all 4 squares fall to the machine!")
        else:
            st.warning("🤝 Draw!")
    else:
        turn = "Your Turn ✍️" if hv_g.current_player == h_pid else "AI Thinking... 🤖"
        pstr = "Red (P1)" if hv_g.current_player == 1 else "Blue (P2)"
        st.info(f"**{turn}** | Current: {pstr} | Move #{hv_g.move_count + 1}")

    col_board, col_side = st.columns([3, 2])
    with col_board:
        fig_hv = draw_qm_board(hv_g.board, hv_g.pool,
                                "Human vs AI — Quantum Match",
                                last_action=last_act,
                                win_flash=hv_g.game_over)
        st.pyplot(fig_hv)
        plt.close(fig_hv)

    with col_side:
        st.markdown("### ⚛️ Board Analysis")
        ti = hv_g.get_threat_info()
        st.metric("Pool Remaining", f"{ti['pool']} tokens")
        st.metric("Empty Squares", ti['empty'])
        st.metric("P1 Controls", f"{ti['p1_owned']} / 4",
                  "⚠️ WIN THREAT!" if ti['p1_can_win'] else "")
        st.metric("P2 Controls", f"{ti['p2_owned']} / 4",
                  "⚠️ WIN THREAT!" if ti['p2_can_win'] else "")
        st.markdown(f"**P1 Threat**: {ti['p1_threat_type']}")
        st.markdown(f"**P2 Threat**: {ti['p2_threat_type']}")

        st.markdown("### 📜 Action Log")
        for i, act in enumerate(hv_g.move_history[-8:]):
            pn = "🔴 P1" if act.player == 1 else "🔵 P2"
            at = "PLACE" if act.action_type == PLACE else "FLIP"
            st.caption(f"Move {i+1}: {pn} {at} → cell {act.position} ({CELL_LABELS[act.position]})")

        if hv_g.event_log:
            st.markdown("### ⚡ Events")
            for ev in hv_g.event_log[-3:]:
                st.caption(ev)

    # Action quality hints
    if not hv_g.game_over and hv_g.current_player == h_pid:
        action_labels = hv_g.get_action_quality_labels()
        valid_actions = hv_g.get_valid_actions()

        st.markdown("---")
        st.markdown("### 🎮 Choose Your Action")

        place_acts = [a for a in valid_actions if a.action_type == PLACE]
        flip_acts = [a for a in valid_actions if a.action_type == FLIP]

        if place_acts:
            st.markdown("**📍 PLACE from pool:**")
            cols = st.columns(min(len(place_acts), 4))
            for ci, act in enumerate(place_acts):
                key = act.to_key()
                ql = action_labels.get(key, "📍 Place")
                btn_lbl = f"{ql}: cell {act.position} ({CELL_LABELS[act.position][:3]})"
                if cols[ci % len(cols)].button(btn_lbl, key=f"qm_p_{act.position}_{hv_g.move_count}"):
                    hv_g.make_action(act)
                    st.session_state.qm_last_action = act
                    st.rerun()

        if flip_acts:
            st.markdown("**🔄 FLIP opponent token:**")
            cols = st.columns(min(len(flip_acts), 4))
            for ci, act in enumerate(flip_acts):
                key = act.to_key()
                ql = action_labels.get(key, "🔄 Flip")
                btn_lbl = f"{ql}: cell {act.position} ({CELL_LABELS[act.position][:3]})"
                if cols[ci % len(cols)].button(btn_lbl, key=f"qm_f_{act.position}_{hv_g.move_count}"):
                    hv_g.make_action(act)
                    st.session_state.qm_last_action = act
                    st.rerun()

    if hv_g.game_over:
        if st.button("🔄 Play Again", use_container_width=True):
            st.session_state.qm_hv_game = QuantumMatchGame()
            st.session_state.qm_hv_active = True
            st.session_state.qm_last_action = None
            st.rerun()

# ============================================================================
# Quick Simulation Mode
# ============================================================================

st.markdown("---")
with st.expander("🔬 Quick AI Simulation"):
    sim_n = st.slider("Games to simulate", 5, 300, 30, 5)
    if st.button("▶️ Run Simulation", key="qm_sim"):
        sim_r = {1: 0, 2: 0, None: 0}
        with st.spinner(f"Running {sim_n} games..."):
            for _ in range(sim_n):
                r = play_qm_game(agent1, agent2, training=False)
                sim_r[r] += 1
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("🔴 P1 Wins", sim_r[1], f"{sim_r[1]/sim_n:.1%}")
        col_s2.metric("🔵 P2 Wins", sim_r[2], f"{sim_r[2]/sim_n:.1%}")
        col_s3.metric("🤝 Draws", sim_r[None], f"{sim_r[None]/sim_n:.1%}")

        # Show last complete game
        demo_g = QuantumMatchGame()
        last_a = None
        while not demo_g.game_over and demo_g.move_count < 40:
            cur = demo_g.current_player
            a = (agent1 if cur == 1 else agent2).choose_action(demo_g, training=False)
            if a is None:
                break
            demo_g.make_action(a)
            last_a = a
        wlbl = f"P{demo_g.winner}" if demo_g.winner else "Draw"
        fig_d = draw_qm_board(demo_g.board, demo_g.pool,
                               f"Last Game — Winner: {wlbl}", last_action=last_a,
                               win_flash=demo_g.game_over)
        st.pyplot(fig_d)
        plt.close(fig_d)

        # Action history of that game
        if demo_g.move_history:
            fig_ah = draw_action_history_chart(demo_g.move_history)
            st.pyplot(fig_ah)
            plt.close(fig_ah)

# ============================================================================
# Rules & Strategy Guide
# ============================================================================

st.markdown("---")
with st.expander("📖 Quantum Match Rules & Strategy Guide"):
    st.markdown("""
## ⚛️ Quantum Match Rules

### The Board
A **2×2 grid** (4 squares total): Top-Left, Top-Right, Bottom-Left, Bottom-Right.

### The Pieces
A shared pool of **4 double-sided tokens** — one side White, one side Black (cosmetically; both players share all tokens).

### On Your Turn
You MUST do exactly one of:
1. **PLACE**: Take one token from the pool and place it on any **empty square**, your color facing up
2. **FLIP**: Take any token showing **your opponent's color** and flip it to **your color**

### Winning Condition
After your turn, if **all 4 squares** show your color — you win immediately.

### Strategic Depth
- ⚡ **Flip pressure**: Once you own 3 squares, you can win by flipping the last one
- 🔮 **Pool control**: Empty squares give placement options; a full board is pure flip warfare
- 🛡️ **Parity attacks**: Prevent your opponent from ever reaching 3 squares simultaneously
- 🧮 **81 states** (3⁴), entirely calculable — but the alternating PLACE/FLIP choice creates rich tactical complexity

### Advanced Strategy
- **Never let opponent reach 3**: If they hold 3 squares, they win by flipping yours on next move
- **Pool depletion**: As pool empties, the game becomes pure flip war — plan ahead
- **Diagonal control**: Controlling opposite corners early limits opponent responses
- **Tempo theft**: Flipping often gains tempo; placing can be passive if not threatening

### AI Architecture
MCTS (planning with PUCT), Negamax with alpha-beta and move ordering, Q-learning table, and a policy table distilled from tree search. Flip/place trade-off analysis is built into both the heuristic evaluator (flip threats, board saturation, mobility) and the prior generation (win detection, opponent blocking, positional value).
""")
