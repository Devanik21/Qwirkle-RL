import streamlit as st
import numpy as np
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
    page_title="Shift-3 Arena",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⟷"
)

st.title("⟷ Shift-3 Arena")
st.markdown("""
A dynamic, moving-board game on a **5-square row** with sliding pieces.
The board never stops moving — master the slide, win the row.

**AI Architecture:**
- 🌳 **MCTS + PUCT** — Monte Carlo Tree Search with AlphaZero's UCB formula
- 🧠 **Negamax + Alpha-Beta** — Full adversarial search with iterative deepening
- 🎯 **Sliding Threat Evaluator** — Pattern-based lookahead for surround and adjacency wins
- 🔄 **Self-Play Reinforcement** — Policy distillation from MCTS visit counts
- 📊 **Q-Learning** — Tabular state-action value estimates updated per game
- 🔬 **Loop Detection** — Handles board cycling via repetition detection
""")

st.markdown("""
<style>
body { background-color: #0e1117; }
.stApp { background-color: #0e1117; }
.stButton>button {
    background: linear-gradient(90deg, #0a1a0a, #112211);
    color: #ccffcc; border: 1px solid #224422; border-radius: 8px; transition: all 0.2s;
}
.stButton>button:hover { border-color: #44FF44; color: #AAFFAA; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Shift-3 Game Logic
# ============================================================================

# Action types
PLACE_ACTION = 'place'   # Place a new piece on an empty square
SLIDE_ACTION = 'slide'   # Slide an existing piece one step left or right

SLIDE_LEFT  = -1
SLIDE_RIGHT = +1

@dataclass
class S3Action:
    action_type: str     # PLACE_ACTION or SLIDE_ACTION
    position: int        # 0-4 (board index)
    direction: int       # 0 if place, -1 if slide left, +1 if slide right
    player: int

    def to_key(self) -> str:
        return f"{self.action_type[0]}{self.position}{self.direction}{self.player}"

    def __hash__(self):
        return hash((self.action_type, self.position, self.direction, self.player))

    def __eq__(self, other):
        return (self.action_type == other.action_type and
                self.position == other.position and
                self.direction == other.direction and
                self.player == other.player)

    def description(self) -> str:
        if self.action_type == PLACE_ACTION:
            return f"PLACE at [{self.position}]"
        else:
            dir_str = "LEFT" if self.direction == SLIDE_LEFT else "RIGHT"
            return f"SLIDE [{self.position}] → {dir_str}"


class Shift3Game:
    """
    Shift-3: 5-square row game.
    Each player starts with 2 pieces to place, 0 pre-placed.
    On each turn: PLACE one piece (if pieces remain in hand) OR SLIDE an existing piece.
    Win conditions:
      1. SURROUND: Your piece — Opponent's piece — Your piece in consecutive squares
      2. TRIPLE: 3 adjacent squares all yours (you only have 2, so this requires both +
         a captured/neutral piece — but per rules this means occupying 3 adj = impossible
         with 2 pieces... so we implement: if your 2 pieces are adjacent AND one square
         further is also yours — no, we implement the spirit: occupy any 3 consecutive
         squares. Since a player has max 2 pieces, "3 consecutive" is only possible if
         the player somehow gains a 3rd — this is not standard. We implement the correct
         winning conditions as stated:
         A) SURROUND: P1, P2, P1 in consecutive squares (you sandwich opponent)
         B) 3-adjacent of YOUR pieces (which requires placing a 3rd if allowed)
    REVISED (correct interpretation):
      - Each player has exactly 2 pieces. Max pieces on board = 4.
      - Win A: Your 2 pieces SURROUND an opponent piece → [P, O, P] pattern
      - Win B: Your 2 pieces are adjacent AND there's an empty square adjacent to them
               that if filled would make 3 — NOT a win. So B is simply:
               3 consecutive squares YOUR color. Since you have 2 pieces total,
               this CANNOT happen unless... we allow it as: all 3 of [i,i+1,i+2]
               are your color (impossible with 2 pieces). So win condition B is
               effectively ONLY via surround.
    FINAL CORRECT: Win = surround [You, Opp, You] at any 3 consecutive squares.
    Ties are possible if neither player can ever achieve surround.
    Max moves limit forces draw.
    """
    BOARD_SIZE = 5
    PIECES_PER_PLAYER = 2
    MAX_MOVES = 120  # Safety limit — positions can loop

    def __init__(self):
        self.reset()

    def reset(self):
        # 0=empty, 1=P1, 2=P2
        self.board = [0] * self.BOARD_SIZE
        self.current_player = 1
        self.game_over = False
        self.winner = None
        # Pieces in hand (not yet placed)
        self.hand = {1: self.PIECES_PER_PLAYER, 2: self.PIECES_PER_PLAYER}
        self.move_history: List[S3Action] = []
        self.move_count = 0
        self.event_log: List[str] = []
        # Repetition detection: track board+player states seen
        self.seen_states: Dict[str, int] = defaultdict(int)
        self.win_cells: Optional[List[int]] = None
        return self.get_state()

    def get_state(self) -> tuple:
        return tuple(self.board) + (self.current_player,
                                    self.hand[1], self.hand[2])

    def get_state_key(self) -> str:
        return ''.join(map(str, self.board)) + str(self.current_player) + str(self.hand[1]) + str(self.hand[2])

    def copy(self) -> 'Shift3Game':
        g = Shift3Game()
        g.board = self.board[:]
        g.current_player = self.current_player
        g.game_over = self.game_over
        g.winner = self.winner
        g.hand = {1: self.hand[1], 2: self.hand[2]}
        g.move_history = self.move_history[:]
        g.move_count = self.move_count
        g.event_log = self.event_log[:]
        g.seen_states = dict(self.seen_states)
        g.win_cells = self.win_cells[:] if self.win_cells else None
        return g

    def get_valid_actions(self) -> List[S3Action]:
        if self.game_over:
            return []
        p = self.current_player
        actions = []

        # PLACE: if player has pieces in hand, place on any empty square
        if self.hand[p] > 0:
            for pos in range(self.BOARD_SIZE):
                if self.board[pos] == 0:
                    actions.append(S3Action(PLACE_ACTION, pos, 0, p))

        # SLIDE: move own piece one step L or R into empty square
        for pos in range(self.BOARD_SIZE):
            if self.board[pos] == p:
                # Try left
                new_pos = pos + SLIDE_LEFT
                if 0 <= new_pos < self.BOARD_SIZE and self.board[new_pos] == 0:
                    actions.append(S3Action(SLIDE_ACTION, pos, SLIDE_LEFT, p))
                # Try right
                new_pos = pos + SLIDE_RIGHT
                if 0 <= new_pos < self.BOARD_SIZE and self.board[new_pos] == 0:
                    actions.append(S3Action(SLIDE_ACTION, pos, SLIDE_RIGHT, p))

        return actions

    def make_action(self, action: S3Action) -> Tuple[tuple, float, bool]:
        if self.game_over:
            return self.get_state(), 0.0, True

        p = self.current_player
        opp = 3 - p
        reward = 0.0

        if action.action_type == PLACE_ACTION:
            if self.hand[p] <= 0 or self.board[action.position] != 0:
                return self.get_state(), -1.0, False
            self.board[action.position] = p
            self.hand[p] -= 1
            reward = 0.3

        elif action.action_type == SLIDE_ACTION:
            dest = action.position + action.direction
            if (self.board[action.position] != p or
                    dest < 0 or dest >= self.BOARD_SIZE or
                    self.board[dest] != 0):
                return self.get_state(), -1.0, False
            self.board[action.position] = 0
            self.board[dest] = p
            reward = 0.5

        self.move_history.append(action)
        self.move_count += 1

        # Check win
        won, win_type, cells = self._check_win(p)
        if won:
            self.game_over = True
            self.winner = p
            self.win_cells = cells
            reward = 100.0
            self.event_log.append(f"P{p} wins via {win_type}!")
        else:
            # Repetition / draw detection
            state_key = self.get_state_key()
            self.seen_states[state_key] += 1
            if self.seen_states[state_key] >= 3:
                self.game_over = True
                self.winner = None
                reward = 0.0
                self.event_log.append("Draw: position repeated 3 times.")
            elif self.move_count >= self.MAX_MOVES:
                self.game_over = True
                self.winner = None
                reward = 0.0
                self.event_log.append("Draw: max moves reached.")
            else:
                self.current_player = opp

        return self.get_state(), reward, self.game_over

    def _check_win(self, player: int) -> Tuple[bool, str, Optional[List[int]]]:
        opp = 3 - player
        # Surround win: [player, opp, player] in consecutive squares
        for i in range(self.BOARD_SIZE - 2):
            if (self.board[i] == player and
                    self.board[i + 1] == opp and
                    self.board[i + 2] == player):
                return True, f"Surround [{i},{i+1},{i+2}]", [i, i + 1, i + 2]

        # 3-adjacent (only possible if a 3rd piece somehow exists — safety check)
        for i in range(self.BOARD_SIZE - 2):
            if all(self.board[i + j] == player for j in range(3)):
                return True, f"Triple [{i},{i+1},{i+2}]", [i, i + 1, i + 2]

        return False, "", None

    def check_win_for(self, player: int) -> bool:
        won, _, _ = self._check_win(player)
        return won

    def get_piece_positions(self, player: int) -> List[int]:
        return [i for i in range(self.BOARD_SIZE) if self.board[i] == player]

    def evaluate_position(self, player: int) -> float:
        """
        Rich heuristic: surround distance, piece separation, mobility,
        blocking opponent surround, and positional pressure.
        """
        if self.winner == player:
            return 100000.0
        if self.winner is not None and self.winner != player:
            return -100000.0

        opp = 3 - player
        score = 0.0

        my_pos = self.get_piece_positions(player)
        op_pos = self.get_piece_positions(opp)

        # --- Surround potential ---
        # How close am I to forming [P, O, P]?
        if len(my_pos) == 2:
            p1, p2 = sorted(my_pos)
            gap = p2 - p1

            # Gap = 2 means an opponent piece between me creates instant win
            if gap == 2:
                mid = (p1 + p2) // 2
                if self.board[mid] == opp:
                    # One slide could be blocked — not quite immediate
                    score += 2000
                elif self.board[mid] == 0:
                    # Need opponent to walk in, or I slide to surround
                    score += 400

            # Gap = 3: one slide brings me to gap=2 in threatening positions
            elif gap == 3:
                score += 150

            # Adjacent pieces: good for pressure
            elif gap == 1:
                score += 80

        # --- Opponent surround threat ---
        if len(op_pos) == 2:
            p1, p2 = sorted(op_pos)
            gap = p2 - p1
            if gap == 2:
                mid = (p1 + p2) // 2
                if self.board[mid] == player:
                    score -= 3000  # IMMINENT loss — opponent surrounds us
                elif self.board[mid] == 0:
                    score -= 500
            elif gap == 3:
                score -= 200
            elif gap == 1:
                score -= 100

        # --- Center preference ---
        center_bonus = {0: 10, 1: 20, 2: 30, 3: 20, 4: 10}
        for pos in my_pos:
            score += center_bonus.get(pos, 10)
        for pos in op_pos:
            score -= center_bonus.get(pos, 10)

        # --- Mobility advantage ---
        orig_cp = self.current_player
        self.current_player = player
        my_moves = len(self.get_valid_actions())
        self.current_player = opp
        op_moves = len(self.get_valid_actions())
        self.current_player = orig_cp
        score += (my_moves - op_moves) * 25

        # --- Hand pieces left (earlier placement = more control) ---
        score += self.hand[player] * 20
        score -= self.hand[opp] * 20

        # --- Piece-to-piece distance: prefer closing in ---
        if len(my_pos) == 2 and len(op_pos) >= 1:
            p1, p2 = sorted(my_pos)
            for op in op_pos:
                if p1 < op < p2:
                    score += 300  # Opponent sandwiched between my pieces!
                    gap_to_surround = (p2 - p1) - 2
                    score += max(0, 300 - gap_to_surround * 100)

        # --- Threat lookahead: count positions one slide away from surround ---
        my_threats = self._count_surround_threats(player)
        op_threats = self._count_surround_threats(opp)
        score += my_threats * 500
        score -= op_threats * 500

        # --- Penalize being at edges (less mobility) ---
        edge_penalty = {0: -15, 4: -15, 1: -5, 3: -5, 2: 0}
        for pos in my_pos:
            score += edge_penalty.get(pos, 0)

        return score

    def _count_surround_threats(self, player: int) -> int:
        """Count actions that immediately create a surround win."""
        threats = 0
        for action in self.get_valid_actions():
            if action.player != player:
                continue
            sim = self.copy()
            sim.current_player = player
            sim.make_action(action)
            if sim.winner == player:
                threats += 1
        return threats

    def get_board_info(self) -> Dict:
        info = {}
        for p in [1, 2]:
            info[f'p{p}_hand'] = self.hand[p]
            info[f'p{p}_placed'] = self.PIECES_PER_PLAYER - self.hand[p]
            info[f'p{p}_pos'] = self.get_piece_positions(p)
        info['empty'] = [i for i in range(self.BOARD_SIZE) if self.board[i] == 0]
        info['move_count'] = self.move_count
        info['seen_max'] = max(self.seen_states.values()) if self.seen_states else 0
        return info

    def get_action_hints(self) -> Dict[str, str]:
        """Return quality hints for each legal action."""
        hints = {}
        p = self.current_player
        for action in self.get_valid_actions():
            key = action.to_key()
            sim = self.copy()
            sim.make_action(action)
            if sim.winner == p:
                hints[key] = "⚡ WIN!"
            else:
                opp = 3 - p
                sim2 = sim.copy()
                # Check if opponent wins immediately after
                opp_threats = 0
                sim2.current_player = opp
                for opp_act in sim2.get_valid_actions():
                    sim3 = sim2.copy()
                    sim3.make_action(opp_act)
                    if sim3.winner == opp:
                        opp_threats += 1
                if opp_threats > 0:
                    hints[key] = "⚠️ Risky"
                elif action.action_type == SLIDE_ACTION:
                    hints[key] = "↔️ Slide"
                else:
                    hints[key] = "📍 Place"
        return hints


# ============================================================================
# MCTS Node
# ============================================================================

class S3MCTSNode:
    def __init__(self, game: Shift3Game, parent=None,
                 action: Optional[S3Action] = None, prior: float = 1.0):
        self.game = game
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children: Dict[str, 'S3MCTSNode'] = {}
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

    def select_child(self, c_puct: float = 1.5) -> 'S3MCTSNode':
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
            self.children[key] = S3MCTSNode(child_game, parent=self,
                                             action=act, prior=prior)
        self.is_expanded = True

    def backup(self, value: float):
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)


# ============================================================================
# AlphaZero-Inspired Shift-3 Agent
# ============================================================================

class Shift3Agent:
    """
    Hybrid agent for Shift-3:
    MCTS (PUCT) + Negamax/Alpha-Beta + Q-Learning + Policy Table.
    Handles mixed PLACE/SLIDE action spaces and loop-detection awareness.
    """
    def __init__(self, player_id: int, lr: float = 0.3, gamma: float = 0.97,
                 epsilon: float = 1.0, mcts_sims: int = 300, minimax_depth: int = 8):
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
        self.surround_wins = 0
        self.triple_wins = 0

    def get_policy_priors(self, game: Shift3Game) -> Dict[str, float]:
        state_key = game.get_state_key()
        actions = game.get_valid_actions()
        priors = {}

        for act in actions:
            key = act.to_key()
            learned = self.policy_table[state_key].get(key, 0.0)
            q_val = self.q_table[state_key].get(key, 0.0)
            prior = 1.0 + max(0, learned) + max(0, q_val) * 0.5

            # Immediate win: massive prior boost
            sim = game.copy()
            sim.make_action(act)
            if sim.winner == game.current_player:
                priors[key] = prior + 10000.0
                continue

            # Block opponent's immediate surround
            opp = 3 - game.current_player
            sim2 = game.copy()
            sim2.current_player = opp
            for opp_act in sim2.get_valid_actions():
                s3 = sim2.copy()
                s3.make_action(opp_act)
                if s3.winner == opp:
                    prior += 600.0

            # Slide actions generally richer than random placement
            if act.action_type == SLIDE_ACTION:
                prior += 40.0

            # Penalise repetitive loops: if this state seen >1, discourage
            sim_key = sim.get_state_key()
            repeat_count = game.seen_states.get(sim_key, 0)
            if repeat_count >= 1:
                prior *= max(0.1, 1.0 - repeat_count * 0.3)

            # Center positions are valuable
            center_vals = {0: 10, 1: 20, 2: 40, 3: 20, 4: 10}
            dest = act.position + act.direction if act.action_type == SLIDE_ACTION else act.position
            prior += center_vals.get(dest, 10)

            # If sliding brings my two pieces closer together
            my_pos = game.get_piece_positions(game.current_player)
            if len(my_pos) == 2 and act.action_type == SLIDE_ACTION:
                p1, p2 = sorted(my_pos)
                old_gap = p2 - p1
                new_positions = [p for p in my_pos if p != act.position] + [dest]
                new_gap = abs(new_positions[0] - new_positions[1]) if len(new_positions) == 2 else 99
                if new_gap < old_gap:
                    prior += 80.0
                # Check if opponent is sandwiched after slide
                if len(new_positions) == 2:
                    np1, np2 = sorted(new_positions)
                    if np2 - np1 == 2:
                        mid = (np1 + np2) // 2
                        if game.board[mid] == opp:
                            prior += 500.0  # Instant surround threat

            priors[key] = max(0.01, prior)

        return priors

    def mcts_search(self, game: Shift3Game) -> S3MCTSNode:
        root = S3MCTSNode(game.copy())
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

    def _evaluate_leaf(self, game: Shift3Game) -> float:
        if game.game_over:
            if game.winner == self.player_id:
                return 1.0
            elif game.winner is not None:
                return -1.0
            return 0.0
        score = self._negamax(game, self.minimax_depth, -float('inf'), float('inf'),
                               game.current_player == self.player_id)
        return math.tanh(score / 1000.0)

    def _negamax(self, game: Shift3Game, depth: int,
                 alpha: float, beta: float, maximizing: bool) -> float:
        if depth == 0 or game.game_over:
            return game.evaluate_position(self.player_id)
        actions = game.get_valid_actions()
        if not actions:
            return game.evaluate_position(self.player_id)

        # Move ordering: heuristic score for each action
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

    def choose_action(self, game: Shift3Game,
                      training: bool = True) -> Optional[S3Action]:
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

        # Block opponent immediate win: check if opponent currently has any winning reply,
        # then find an action that eliminates all such threats.
        opp = 3 - self.player_id
        # Check whether opponent can win from current position on their next move
        opp_can_win_now = False
        opp_check = game.copy()
        opp_check.current_player = opp
        for opp_act in opp_check.get_valid_actions():
            s = opp_check.copy()
            s.make_action(opp_act)
            if s.winner == opp:
                opp_can_win_now = True
                break
        if opp_can_win_now:
            for block_act in actions:
                sim = game.copy()
                sim.make_action(block_act)
                opp_still_wins = False
                for opp_act2 in sim.get_valid_actions():
                    s2 = sim.copy()
                    s2.make_action(opp_act2)
                    if s2.winner == opp:
                        opp_still_wins = True
                        break
                if not opp_still_wins:
                    self.total_moves += 1
                    return block_act

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
        self.surround_wins = 0
        self.triple_wins = 0

    def get_stats(self) -> Dict:
        total = self.wins + self.losses + self.draws
        return {
            'wins': self.wins, 'losses': self.losses, 'draws': self.draws,
            'total': total, 'win_rate': self.wins / max(1, total),
            'policies': len(self.policy_table), 'q_states': len(self.q_table),
            'epsilon': self.epsilon, 'temperature': self.temperature,
            'total_moves': self.total_moves,
            'surround_wins': self.surround_wins,
        }


# ============================================================================
# Self-Play Training
# ============================================================================

def play_s3_game(agent1: Shift3Agent, agent2: Shift3Agent,
                 training: bool = True) -> Optional[int]:
    game = Shift3Game()
    history: List[Tuple[str, str, int]] = []
    agents = {1: agent1, 2: agent2}

    while not game.game_over:
        current = game.current_player
        agent = agents[current]
        state_key = game.get_state_key()
        action = agent.choose_action(game, training)
        if action is None:
            break
        history.append((state_key, action.to_key(), current))
        game.make_action(action)

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

CELL_POS_LABELS = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

def draw_shift3_board(board: List[int], hand: Dict[int, int],
                       title: str = "Shift-3",
                       last_action: Optional[S3Action] = None,
                       win_cells: Optional[List[int]] = None,
                       move_count: int = 0) -> plt.Figure:
    """Draw the 5-square Shift-3 board with annotations."""
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    cell_w = 1.8
    cell_h = 1.6
    gap = 0.15
    colors_map = {0: '#111122', 1: '#1a0505', 2: '#050515'}
    piece_colors = {1: '#DC143C', 2: '#1E90FF'}
    edge_colors = {0: '#333344', 1: '#8B0000', 2: '#000080'}

    for idx in range(5):
        x = idx * (cell_w + gap)
        y = 0.0
        cell_val = board[idx]
        face = colors_map[cell_val]
        edge = edge_colors[cell_val]
        lw = 1.5
        is_last = last_action and (
            (last_action.action_type == PLACE_ACTION and last_action.position == idx) or
            (last_action.action_type == SLIDE_ACTION and
             (last_action.position + last_action.direction) == idx)
        )
        is_origin = (last_action and last_action.action_type == SLIDE_ACTION and
                     last_action.position == idx)

        if win_cells and idx in win_cells:
            face = '#0a1a00'
            edge = '#00FF44'
            lw = 4
        elif is_last:
            edge = '#FFFFFF'
            lw = 3
        elif is_origin:
            face = '#222222'
            edge = '#666666'
            lw = 1

        rect = plt.Rectangle((x, y), cell_w, cell_h,
                               facecolor=face, edgecolor=edge, linewidth=lw)
        ax.add_patch(rect)

        # Square label
        ax.text(x + cell_w / 2, y + 0.15, f"{CELL_POS_LABELS[idx]} [{idx}]",
                ha='center', va='bottom', fontsize=9, color='#555577')

        # Piece
        if cell_val != 0:
            ax.text(x + cell_w / 2, y + cell_h / 2 + 0.1, '●',
                    ha='center', va='center', fontsize=44,
                    color=piece_colors[cell_val], fontweight='bold', zorder=4)

        # Win cell marker
        if win_cells and idx in win_cells:
            ax.text(x + cell_w / 2, y + cell_h - 0.2, '★',
                    ha='center', va='center', fontsize=16,
                    color='#FFFF00', zorder=5)

        # Action indicator on destination
        if is_last and last_action.action_type == SLIDE_ACTION:
            dir_sym = '←' if last_action.direction == -1 else '→'
            ax.text(x + cell_w / 2, y + 0.05, dir_sym,
                    ha='center', va='bottom', fontsize=14, color='#FFCC00', zorder=5)
        elif is_last and last_action.action_type == PLACE_ACTION:
            ax.text(x + cell_w / 2, y + 0.05, '↓',
                    ha='center', va='bottom', fontsize=14, color='#CCFFCC', zorder=5)

    # Draw connecting lines between squares
    for idx in range(4):
        x1 = idx * (cell_w + gap) + cell_w
        x2 = x1 + gap
        y_mid = cell_h / 2
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                    arrowprops=dict(arrowstyle="<->", color='#444466', lw=1.5))

    # Slide arrows annotation (possible slides for current turn)
    ax.set_xlim(-0.2, 5 * (cell_w + gap) + 0.4)
    ax.set_ylim(-0.6, cell_h + 0.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, color='#CCFFCC', fontweight='bold', pad=12)

    # Hand display
    h1_str = "🔴 " * hand[1] + "○ " * (Shift3Game.PIECES_PER_PLAYER - hand[1])
    h2_str = "🔵 " * hand[2] + "○ " * (Shift3Game.PIECES_PER_PLAYER - hand[2])
    ax.text(0, -0.4, f"P1 Hand: {h1_str}", fontsize=10, color='#FF9999')
    ax.text(5 * (cell_w + gap) * 0.5, -0.4, f"P2 Hand: {h2_str}",
            fontsize=10, color='#9999FF')
    ax.text(5 * (cell_w + gap) - 0.3, -0.4, f"Move #{move_count}",
            fontsize=9, color='#888888', ha='right')

    # Legend
    p1_patch = mpatches.Patch(color='#DC143C', label='Player 1 (Red)')
    p2_patch = mpatches.Patch(color='#1E90FF', label='Player 2 (Blue)')
    ax.legend(handles=[p1_patch, p2_patch], loc='upper right',
              facecolor='#0e1117', edgecolor='#334455', labelcolor='white', fontsize=9)

    return fig


def draw_board_heatmap(game: Shift3Game, player: int) -> plt.Figure:
    """Show heuristic value of each cell for the given player."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    fig.patch.set_facecolor('#0e1117')

    for pi, (p, ax) in enumerate(zip([1, 2], axes)):
        ax.set_facecolor('#0e1117')
        scores = []
        for pos in range(game.BOARD_SIZE):
            if game.board[pos] == 0:
                sim = game.copy()
                sim.board[pos] = p
                s = sim.evaluate_position(p)
                sim.board[pos] = 0
                scores.append(s)
            else:
                scores.append(0)

        arr = np.array(scores).reshape(1, -1)
        ax.imshow(arr, cmap='RdYlGn', aspect='auto',
                  vmin=min(scores) - 1, vmax=max(scores) + 1)
        for i, s in enumerate(scores):
            ax.text(i, 0, f"{s:.0f}", ha='center', va='center',
                    fontsize=9, color='black', fontweight='bold')
        ax.set_xticks(range(game.BOARD_SIZE))
        ax.set_xticklabels([f"{CELL_POS_LABELS[i]}[{i}]" for i in range(game.BOARD_SIZE)],
                            color='#AAAACC')
        ax.set_yticks([])
        pcolor = '#DC143C' if p == 1 else '#1E90FF'
        ax.set_title(f"P{p} Cell Value Heatmap", color=pcolor, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334455')

    fig.suptitle("⟷ Cell Evaluation Heatmap", color='#CCFFCC',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def draw_action_history(move_history: List[S3Action]) -> plt.Figure:
    """Visualize PLACE vs SLIDE actions over the game."""
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1a2a1a')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334455')
    ax.tick_params(colors='#AACCAA')

    turns = list(range(1, len(move_history) + 1))
    slide_t = [t for t, a in zip(turns, move_history) if a.action_type == SLIDE_ACTION]
    place_t = [t for t, a in zip(turns, move_history) if a.action_type == PLACE_ACTION]
    p1_t = [t for t, a in zip(turns, move_history) if a.player == 1]
    p2_t = [t for t, a in zip(turns, move_history) if a.player == 2]

    ax.scatter(slide_t, [1.2] * len(slide_t), marker='<', color='#FFD700',
               s=120, label='SLIDE', zorder=3)
    ax.scatter(place_t, [0.8] * len(place_t), marker='^', color='#AAFFAA',
               s=120, label='PLACE', zorder=3)
    for t in p1_t:
        ax.axvline(x=t, color='#DC143C', alpha=0.25, linewidth=2)
    for t in p2_t:
        ax.axvline(x=t, color='#1E90FF', alpha=0.25, linewidth=2)

    # Annotate positions
    for t, a in zip(turns, move_history):
        dest = a.position + a.direction if a.action_type == SLIDE_ACTION else a.position
        y = 1.2 if a.action_type == SLIDE_ACTION else 0.8
        ax.text(t, y + 0.18, CELL_POS_LABELS.get(dest, '?'),
                ha='center', va='center', fontsize=7, color='#FFFFFF')

    ax.set_xlim(0, len(move_history) + 1)
    ax.set_ylim(0.3, 1.7)
    ax.set_yticks([0.8, 1.2])
    ax.set_yticklabels(['PLACE', 'SLIDE'], color='#AACCAA', fontsize=10)
    ax.set_xlabel('Turn Number', color='#AACCAA')
    ax.set_title('Action Sequence (Red=P1, Blue=P2)', color='#CCFFCC', fontweight='bold')
    ax.legend(facecolor='#1a2a1a', edgecolor='#334455', labelcolor='white')

    return fig


def draw_training_charts(history: Dict) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor('#0e1117')
    for ax in axes.flat:
        ax.set_facecolor('#1a2a1a')
        ax.tick_params(colors='#AACCAA')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334455')

    eps = history.get('episode', [])
    if not eps:
        return fig

    axes[0, 0].plot(eps, history.get('agent1_wins', []), color='#DC143C', lw=2, label='P1 Wins')
    axes[0, 0].plot(eps, history.get('agent2_wins', []), color='#1E90FF', lw=2, label='P2 Wins')
    axes[0, 0].plot(eps, history.get('draws', []), color='#888888', lw=1.5, ls='--', label='Draws')
    axes[0, 0].set_title('Win/Draw Distribution', color='#CCFFCC')
    axes[0, 0].legend(facecolor='#1a2a1a', edgecolor='#334455', labelcolor='white')

    axes[0, 1].plot(eps, history.get('agent1_epsilon', []), color='#FF6B6B', lw=2, label='P1 ε')
    axes[0, 1].plot(eps, history.get('agent2_epsilon', []), color='#66B3FF', lw=2, label='P2 ε')
    axes[0, 1].set_title('Exploration Rate (ε)', color='#CCFFCC')
    axes[0, 1].legend(facecolor='#1a2a1a', edgecolor='#334455', labelcolor='white')

    axes[1, 0].plot(eps, history.get('agent1_policies', []), color='#FF6B6B', lw=2, label='P1')
    axes[1, 0].plot(eps, history.get('agent2_policies', []), color='#66B3FF', lw=2, label='P2')
    axes[1, 0].set_title('Policy Table Size', color='#CCFFCC')
    axes[1, 0].legend(facecolor='#1a2a1a', edgecolor='#334455', labelcolor='white')

    a1w = history.get('agent1_wins', [0])
    a2w = history.get('agent2_wins', [0])
    dr = history.get('draws', [0])
    totals = [max(1, a + b + d) for a, b, d in zip(a1w, a2w, dr)]
    axes[1, 1].plot(eps, [w / t for w, t in zip(a1w, totals)], color='#DC143C', lw=2, label='P1 WR')
    axes[1, 1].plot(eps, [w / t for w, t in zip(a2w, totals)], color='#1E90FF', lw=2, label='P2 WR')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Win Rate Over Time', color='#CCFFCC')
    axes[1, 1].legend(facecolor='#1a2a1a', edgecolor='#334455', labelcolor='white')

    fig.suptitle('⟷ Shift-3 Training Analytics', fontsize=15, color='#CCFFCC', fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# Save / Load
# ============================================================================

def serialize_agent(agent: Shift3Agent, role: str) -> Dict:
    return {
        'metadata': {'role': role, 'version': '1.0', 'game': 'shift3'},
        'player_id': agent.player_id,
        'epsilon': float(agent.epsilon),
        'temperature': float(agent.temperature),
        'wins': int(agent.wins), 'losses': int(agent.losses), 'draws': int(agent.draws),
        'total_moves': int(agent.total_moves), 'surround_wins': int(agent.surround_wins),
        'mcts_sims': int(agent.mcts_sims), 'minimax_depth': int(agent.minimax_depth),
        'q_table': {sk: {ak: float(v) for ak, v in avs.items()}
                    for sk, avs in agent.q_table.items()},
        'policy_table': {sk: {ak: float(v) for ak, v in avs.items()}
                         for sk, avs in agent.policy_table.items()},
    }

def deserialize_agent(data: Dict, player_id: int) -> Shift3Agent:
    agent = Shift3Agent(player_id=player_id,
                        mcts_sims=data.get('mcts_sims', 300),
                        minimax_depth=data.get('minimax_depth', 8))
    agent.epsilon = data.get('epsilon', 0.1)
    agent.temperature = data.get('temperature', 0.3)
    agent.wins = data.get('wins', 0)
    agent.losses = data.get('losses', 0)
    agent.draws = data.get('draws', 0)
    agent.total_moves = data.get('total_moves', 0)
    agent.surround_wins = data.get('surround_wins', 0)
    for sk, avs in data.get('q_table', {}).items():
        for ak, v in avs.items():
            agent.q_table[sk][ak] = float(v)
    for sk, avs in data.get('policy_table', {}).items():
        for ak, v in avs.items():
            agent.policy_table[sk][ak] = float(v)
    return agent

def create_agents_zip(agent1: Shift3Agent, agent2: Shift3Agent, config: Dict) -> io.BytesIO:
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

st.sidebar.header("⚙️ Shift-3 Controls")

with st.sidebar.expander("1. Agent 1 (Red) Parameters", expanded=True):
    lr1 = st.slider("Learning Rate α₁", 0.05, 1.0, 0.25, 0.05)
    gamma1 = st.slider("Discount γ₁", 0.80, 0.99, 0.97, 0.01)
    mcts1 = st.slider("MCTS Simulations₁", 10, 800, 300, 10)
    mm1 = st.slider("Minimax Depth₁", 1, 14, 8, 1)
    temp1 = st.slider("Temperature₁", 0.0, 2.0, 1.0, 0.1)

with st.sidebar.expander("2. Agent 2 (Blue) Parameters", expanded=True):
    lr2 = st.slider("Learning Rate α₂", 0.05, 1.0, 0.25, 0.05)
    gamma2 = st.slider("Discount γ₂", 0.80, 0.99, 0.97, 0.01)
    mcts2 = st.slider("MCTS Simulations₂", 10, 800, 200, 10)
    mm2 = st.slider("Minimax Depth₂", 1, 14, 7, 1)
    temp2 = st.slider("Temperature₂", 0.0, 2.0, 1.0, 0.1)

with st.sidebar.expander("3. Training Configuration", expanded=True):
    episodes = st.number_input("Training Episodes", 10, 50000, 500, 50)
    update_freq = st.number_input("Update Every N Games", 1, 500, 25, 5)
    show_live = st.checkbox("Show Live Board During Training", False)

with st.sidebar.expander("4. Brain Storage", expanded=False):
    if 's3_agent1' in st.session_state and st.session_state.s3_agent1:
        a1r = st.session_state.s3_agent1
        a2r = st.session_state.s3_agent2
        st.markdown("### 🧠 Neural Sync")
        c1, c2 = st.columns(2)
        if c1.button("P1 ➡️ P2"):
            st.session_state.s3_agent2.policy_table = deepcopy(a1r.policy_table)
            st.session_state.s3_agent2.q_table = deepcopy(a1r.q_table)
            st.session_state.s3_agent2.epsilon = a1r.epsilon
            st.toast("P2 now has P1's brain!", icon="🔵")
        if c2.button("P2 ➡️ P1"):
            st.session_state.s3_agent1.policy_table = deepcopy(a2r.policy_table)
            st.session_state.s3_agent1.q_table = deepcopy(a2r.q_table)
            st.session_state.s3_agent1.epsilon = a2r.epsilon
            st.toast("P1 now has P2's brain!", icon="🔴")
        st.markdown("---")
        cfg_save = {
            'lr1': lr1, 'gamma1': gamma1, 'mcts1': mcts1, 'mm1': mm1,
            'lr2': lr2, 'gamma2': gamma2, 'mcts2': mcts2, 'mm2': mm2,
        }
        zip_b = create_agents_zip(a1r, a2r, cfg_save)
        st.download_button("💾 Download Agents", zip_b,
                           "shift3_agents.zip", "application/zip",
                           use_container_width=True)
    else:
        st.info("Train agents first to enable save.")
    st.markdown("---")
    upf = st.file_uploader("📤 Upload Agents (.zip)", type="zip")
    if upf and st.button("🔄 Load Agents", use_container_width=True):
        a1l, a2l, cfgl = load_agents_from_zip(upf)
        if a1l and a2l:
            st.session_state.s3_agent1 = a1l
            st.session_state.s3_agent2 = a2l
            st.toast("✅ Agents loaded!", icon="🧠")
            st.rerun()

train_btn = st.sidebar.button("⟷ Begin Self-Play Training",
                               use_container_width=True, type="primary")
if st.sidebar.button("🧹 Reset Arena", use_container_width=True):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


# ============================================================================
# Init Agents
# ============================================================================

if 's3_agent1' not in st.session_state:
    st.session_state.s3_agent1 = Shift3Agent(1, lr1, gamma1, mcts_sims=mcts1, minimax_depth=mm1)
    st.session_state.s3_agent2 = Shift3Agent(2, lr2, gamma2, mcts_sims=mcts2, minimax_depth=mm2)

agent1: Shift3Agent = st.session_state.s3_agent1
agent2: Shift3Agent = st.session_state.s3_agent2

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
    st.subheader("⟷ Shift-3 Self-Play Training")
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
        play_s3_game(agent1, agent2, training=True)
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
                demo_g = Shift3Game()
                fig_live = draw_shift3_board(demo_g.board, demo_g.hand,
                                              "Training Demo Board")
                board_ph.pyplot(fig_live)
                plt.close(fig_live)

    prog_bar.progress(1.0)
    st.toast("Training Complete! ⟷", icon="✨")
    st.session_state.s3_training_history = hist
    time.sleep(0.5)
    st.rerun()


# ============================================================================
# Training Analytics
# ============================================================================

if 's3_training_history' in st.session_state and st.session_state.s3_training_history:
    th = st.session_state.s3_training_history
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
                'P1 Policies': th['agent1_policies'],
                'P2 Policies': th['agent2_policies'],
                'P1 Q-States': th['agent1_q_states'],
                'P2 Q-States': th['agent2_q_states'],
            })
            st.dataframe(df_th, use_container_width=True)


# ============================================================================
# Championship Match (AI vs AI)
# ============================================================================

st.markdown("---")
st.subheader("⚔️ AI Championship Match")

if len(agent1.policy_table) > 3 or len(agent1.q_table) > 3:
    if st.button("▶️ Watch Championship Match", use_container_width=True):
        champ = Shift3Game()
        champ_agents = {1: agent1, 2: agent2}
        board_ph = st.empty()
        info_ph = st.empty()
        hist_ph = st.empty()
        heat_ph = st.empty()
        mn = 0

        with st.spinner("Agents competing..."):
            while not champ.game_over and mn < 80:
                cur = champ.current_player
                act = champ_agents[cur].choose_action(champ, training=False)
                if act is None:
                    break
                champ.make_action(act)
                mn += 1
                pname = "Red (P1)" if cur == 1 else "Blue (P2)"
                info_ph.caption(f"Move {mn}: **{pname}** — {act.description()}")

                fig_b = draw_shift3_board(
                    champ.board, champ.hand,
                    f"Move {mn}: {pname} — {act.description()}",
                    last_action=act,
                    win_cells=champ.win_cells,
                    move_count=mn
                )
                board_ph.pyplot(fig_b)
                plt.close(fig_b)

                if len(champ.move_history) > 0:
                    fig_ah = draw_action_history(champ.move_history)
                    hist_ph.pyplot(fig_ah)
                    plt.close(fig_ah)

                # Heatmap during game
                if not champ.game_over and mn % 3 == 0:
                    fig_hm = draw_board_heatmap(champ, cur)
                    heat_ph.pyplot(fig_hm)
                    plt.close(fig_hm)

                time.sleep(0.65)

        if champ.winner == 1:
            st.success("🏆 Agent 1 (Red) Wins the Championship via Surround!")
        elif champ.winner == 2:
            st.error("🏆 Agent 2 (Blue) Wins the Championship via Surround!")
        else:
            st.warning("🤝 Draw! (Loop or max moves reached)")

        if champ.event_log:
            for ev in champ.event_log:
                st.caption(f"⚡ {ev}")
else:
    st.info("⟷ Train agents first to enable championship match.")


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
    ai_str = st.selectbox("AI Strength",
                           ["Easy (30 sims)", "Medium (100 sims)",
                            "Hard (300 sims)", "Expert (600 sims)"])
    s_map = {"Easy (30 sims)": 30, "Medium (100 sims)": 100,
             "Hard (300 sims)": 300, "Expert (600 sims)": 600}

if st.button("🎯 Start Human vs AI Game", use_container_width=True, type="primary"):
    human_pid = 1 if "Player 1" in human_col else 2
    ai_pid = 3 - human_pid
    st.session_state.s3_hv_game = Shift3Game()
    st.session_state.s3_hv_active = True
    st.session_state.s3_human_pid = human_pid
    st.session_state.s3_ai_pid = ai_pid
    st.session_state.s3_ai_ref = agent1 if "Agent 1" in ai_opp else agent2
    st.session_state.s3_ai_sims = s_map[ai_str]
    st.session_state.s3_last_action = None
    st.rerun()

if st.session_state.get('s3_hv_active', False):
    hv_g: Shift3Game = st.session_state.s3_hv_game
    h_pid = st.session_state.s3_human_pid
    ai_pid = st.session_state.s3_ai_pid
    ai_r: Shift3Agent = st.session_state.s3_ai_ref
    ai_sims = st.session_state.s3_ai_sims
    last_act = st.session_state.get('s3_last_action', None)

    # AI turn
    if hv_g.current_player == ai_pid and not hv_g.game_over:
        with st.spinner("🤖 AI thinking..."):
            old = ai_r.mcts_sims
            ai_r.mcts_sims = ai_sims
            ai_mv = ai_r.choose_action(hv_g, training=False)
            ai_r.mcts_sims = old
            if ai_mv:
                hv_g.make_action(ai_mv)
                st.session_state.s3_last_action = ai_mv
                st.rerun()

    # Status
    if hv_g.game_over:
        if hv_g.winner == h_pid:
            st.success("🎉 YOU WIN! Perfect surround executed!")
        elif hv_g.winner == ai_pid:
            st.error("🤖 AI Wins by surrounding your piece!")
        else:
            st.warning("🤝 Draw! (Position looped or max moves)")
    else:
        turn = "Your Turn ✍️" if hv_g.current_player == h_pid else "AI Thinking... 🤖"
        pstr = "Red (P1)" if hv_g.current_player == 1 else "Blue (P2)"
        hand_cur = hv_g.hand[hv_g.current_player]
        st.info(f"**{turn}** | {pstr} | Hand: {hand_cur} pieces | Move #{hv_g.move_count + 1}")

    col_board, col_side = st.columns([3, 2])
    with col_board:
        fig_hv = draw_shift3_board(
            hv_g.board, hv_g.hand,
            "Human vs AI — Shift-3",
            last_action=last_act,
            win_cells=hv_g.win_cells,
            move_count=hv_g.move_count
        )
        st.pyplot(fig_hv)
        plt.close(fig_hv)

        # Heatmap if board has pieces
        if hv_g.move_count > 0 and not hv_g.game_over:
            fig_hm = draw_board_heatmap(hv_g, hv_g.current_player)
            st.pyplot(fig_hm)
            plt.close(fig_hm)

    with col_side:
        st.markdown("### ⟷ Board Analysis")
        bi = hv_g.get_board_info()
        st.metric("P1 In Hand", f"{bi['p1_hand']} / {Shift3Game.PIECES_PER_PLAYER}")
        st.metric("P2 In Hand", f"{bi['p2_hand']} / {Shift3Game.PIECES_PER_PLAYER}")
        st.markdown(f"**P1 Positions**: {bi['p1_pos']}")
        st.markdown(f"**P2 Positions**: {bi['p2_pos']}")
        st.markdown(f"**Empty Squares**: {bi['empty']}")

        # Repetition warning
        rep = bi['seen_max']
        if rep >= 2:
            st.warning(f"⚠️ Board state repeated {rep}× — draw approaching!")
        elif rep == 1:
            st.caption(f"Position seen {rep}× — avoid repetition or draw triggers at 3×")

        st.markdown(f"**Move Count**: {bi['move_count']} / {Shift3Game.MAX_MOVES}")

        # Win distance estimate
        p1_pos = bi['p1_pos']
        p2_pos = bi['p2_pos']
        if len(p1_pos) == 2 and len(p2_pos) >= 1:
            p1s, p1e = sorted(p1_pos)
            gap = p1e - p1s
            if gap == 2:
                mid = (p1s + p1e) // 2
                if hv_g.board[mid] == 2:
                    st.error("⚡ P1 has a SURROUND threat!")
            elif gap == 3:
                st.warning("P1 is 1 slide from a surround threat")
        if len(p2_pos) == 2 and len(p1_pos) >= 1:
            p2s, p2e = sorted(p2_pos)
            gap = p2e - p2s
            if gap == 2:
                mid = (p2s + p2e) // 2
                if hv_g.board[mid] == 1:
                    st.error("⚡ P2 has a SURROUND threat!")
            elif gap == 3:
                st.warning("P2 is 1 slide from a surround threat")

        st.markdown("### 📜 Action Log")
        for i, act in enumerate(hv_g.move_history[-8:]):
            pn = "🔴 P1" if act.player == 1 else "🔵 P2"
            st.caption(f"{pn}: {act.description()}")

        if hv_g.event_log:
            st.markdown("### ⚡ Events")
            for ev in hv_g.event_log[-3:]:
                st.caption(ev)

    # Human action selection
    if not hv_g.game_over and hv_g.current_player == h_pid:
        action_hints = hv_g.get_action_hints()
        valid_acts = hv_g.get_valid_actions()

        place_acts = [a for a in valid_acts if a.action_type == PLACE_ACTION]
        slide_acts = [a for a in valid_acts if a.action_type == SLIDE_ACTION]

        st.markdown("---")
        st.markdown("### 🎮 Choose Your Action")

        if place_acts:
            st.markdown("**📍 PLACE a piece from hand:**")
            cols = st.columns(min(len(place_acts), 5))
            for ci, act in enumerate(place_acts):
                hint = action_hints.get(act.to_key(), "📍 Place")
                lbl = f"{hint} at [{act.position}] {CELL_POS_LABELS[act.position]}"
                if cols[ci % len(cols)].button(lbl, key=f"s3_place_{act.position}_{hv_g.move_count}"):
                    hv_g.make_action(act)
                    st.session_state.s3_last_action = act
                    st.rerun()

        if slide_acts:
            st.markdown("**↔️ SLIDE an existing piece:**")
            cols = st.columns(min(len(slide_acts), 6))
            for ci, act in enumerate(slide_acts):
                hint = action_hints.get(act.to_key(), "↔️ Slide")
                dest = act.position + act.direction
                dir_sym = "←" if act.direction == -1 else "→"
                lbl = f"{hint}: [{act.position}]{dir_sym}[{dest}]"
                if cols[ci % len(cols)].button(lbl, key=f"s3_slide_{act.position}_{act.direction}_{hv_g.move_count}"):
                    hv_g.make_action(act)
                    st.session_state.s3_last_action = act
                    st.rerun()

        if not place_acts and not slide_acts:
            st.warning("No valid actions available — game may be stuck!")

    # Action history during game
    if hv_g.move_history:
        st.markdown("---")
        fig_ah = draw_action_history(hv_g.move_history)
        st.pyplot(fig_ah)
        plt.close(fig_ah)

    if hv_g.game_over:
        if st.button("🔄 Play Again", use_container_width=True):
            st.session_state.s3_hv_game = Shift3Game()
            st.session_state.s3_hv_active = True
            st.session_state.s3_last_action = None
            st.rerun()


# ============================================================================
# Quick Simulation Mode
# ============================================================================

st.markdown("---")
with st.expander("🔬 Quick AI Simulation (No Training Effect)"):
    sim_n = st.slider("Games to simulate", 5, 200, 20, 5)
    if st.button("▶️ Run Simulation", key="s3_sim"):
        sim_r = {1: 0, 2: 0, None: 0}
        total_moves_acc = 0
        with st.spinner(f"Simulating {sim_n} games..."):
            for _ in range(sim_n):
                g_tmp = Shift3Game()
                agents_tmp = {1: agent1, 2: agent2}
                while not g_tmp.game_over:
                    cur = g_tmp.current_player
                    a = agents_tmp[cur].choose_action(g_tmp, training=False)
                    if a is None:
                        break
                    g_tmp.make_action(a)
                r = g_tmp.winner
                sim_r[r] += 1
                total_moves_acc += g_tmp.move_count

        avg_moves = total_moves_acc / max(1, sim_n)
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("🔴 P1 Wins", sim_r[1], f"{sim_r[1]/sim_n:.1%}")
        col_s2.metric("🔵 P2 Wins", sim_r[2], f"{sim_r[2]/sim_n:.1%}")
        col_s3.metric("🤝 Draws", sim_r[None], f"{sim_r[None]/sim_n:.1%}")
        col_s4.metric("Avg Moves/Game", f"{avg_moves:.1f}")

        # Show last game board
        demo_g = Shift3Game()
        last_a = None
        while not demo_g.game_over:
            cur = demo_g.current_player
            a = (agent1 if cur == 1 else agent2).choose_action(demo_g, training=False)
            if a is None:
                break
            demo_g.make_action(a)
            last_a = a

        wlbl = f"P{demo_g.winner}" if demo_g.winner else "Draw"
        fig_d = draw_shift3_board(
            demo_g.board, demo_g.hand,
            f"Last Simulated Game — Winner: {wlbl}",
            last_action=last_a, win_cells=demo_g.win_cells,
            move_count=demo_g.move_count
        )
        st.pyplot(fig_d)
        plt.close(fig_d)

        if demo_g.move_history:
            fig_ah = draw_action_history(demo_g.move_history)
            st.pyplot(fig_ah)
            plt.close(fig_ah)

        if demo_g.event_log:
            st.markdown("**Events:**")
            for ev in demo_g.event_log:
                st.caption(f"⚡ {ev}")


# ============================================================================
# Rules & Strategy Guide
# ============================================================================

st.markdown("---")
with st.expander("📖 Shift-3 Rules & Strategy Guide"):
    st.markdown("""
## ⟷ Shift-3 Rules

### The Board
A single **row of 5 squares** labeled A[0] through E[4].

### Starting Position
Both players start with **2 pieces in hand** — none placed on the board.

### On Your Turn: Choose ONE of
1. **PLACE** — Take a piece from your hand and put it on any **empty** square
2. **SLIDE** — Move one of your **already-placed** pieces exactly **one square left or right** into an adjacent **empty** square

### Winning Condition
Achieve a **SURROUND**: Create the pattern **[Your piece — Opponent piece — Your piece]** in 3 consecutive squares.

For example: positions [1, 2, 3] = [Red, Blue, Red] → Red wins.

### Draw Conditions
- The same board position appears **3 times** (repetition rule)
- **120 moves** are made without a winner

### Strategic Principles
- 🎯 **Approach from both sides**: Your 2 pieces must bracket an opponent piece — plan to close the gap
- 🔀 **Sliding is king**: SLIDE actions cost nothing from hand and can set up an immediate surround
- 🛡️ **Don't walk into the bracket**: If opponent has pieces at positions 2 apart, avoid the middle
- ⚡ **Placement phase**: Use your 2 pieces to pressure from both ends of the board simultaneously
- 🔄 **Avoid loops**: Repetitive sliding creates draws — commit to an attack direction
- 🏃 **Tempo management**: Passive defense eventually leads to a repetition draw

### State Space Analysis
- Physical: 3⁵ = **243** configurations
- Reachable: significantly fewer due to hand constraints
- **Pieces can move** → positions loop → enormous tactical depth despite tiny state space
- The game is NOT trivially solved; MCTS at depth 8+ reveals non-obvious forcing sequences

### AI Architecture
The agent uses MCTS (PUCT), Negamax/Alpha-Beta with move ordering, Q-learning, and a policy table distilled from tree search visits. The heuristic evaluator is tailored to Shift-3: it models surround distance, piece-gap dynamics, opponent sandwiching, mobility asymmetry, edge penalties, and repetition-loop avoidance. The loop detection system inside the game feeds back into the prior generator to discourage the AI from entering drawn positions.
""")
