import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from collections import deque, defaultdict
import random
import pandas as pd
import json
import zipfile
import io
from copy import deepcopy
import math

# ============================================================================
# Page Config and Initial Setup
# ============================================================================
st.set_page_config(
    page_title="Strategic RL Qwirkle",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎨"
)

st.title("Strategic Qwirkle RL Arena")
st.markdown("""
Watch two Reinforcement Learning agents master the strategic tile-laying game of **Qwirkle** through intelligent learning and MCTS planning.

**🎯 Qwirkle Rules:**
- 108 tiles: 6 colors × 6 shapes (3 of each combo)
- Create lines of same color OR same shape (no duplicates)
- Score 1 point per tile in lines you create/extend
- **QWIRKLE BONUS:** Complete a line of 6 tiles = 12 points (6 + 6 bonus)
- Tiles can score in multiple lines simultaneously

**Core Algorithmic Components:**
- 🌳 **Monte Carlo Tree Search (MCTS)** - Strategic simulation & planning
- 🧮 **UCT (Upper Confidence Bound)** - Optimal exploration-exploitation
- 🎓 **Q-Learning with experience replay** - Pattern recognition
- 📊 **Advanced position evaluation** - Multi-factor board analysis
- 🎯 **Strategic tile placement** - Maximizing Qwirkle opportunities
""")

# ============================================================================
# Game Constants
# ============================================================================
COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
SHAPES = ['circle', 'square', 'diamond', 'star', 'clover', 'cross']
COLOR_CODES = {
    'red': '#e74c3c', 'orange': '#e67e22', 'yellow': '#f39c12',
    'green': '#27ae60', 'blue': '#3498db', 'purple': '#9b59b6'
}
SHAPE_SYMBOLS = {
    'circle': '●', 'square': '■', 'diamond': '◆',
    'star': '★', 'clover': '♣', 'cross': '✚'
}

# ============================================================================
# Qwirkle Game Environment
# ============================================================================

class Tile:
    def __init__(self, color, shape):
        self.color = color
        self.shape = shape
    
    def __eq__(self, other):
        return self.color == other.color and self.shape == other.shape
    
    def __hash__(self):
        return hash((self.color, self.shape))
    
    def __repr__(self):
        return f"{self.color[0].upper()}{self.shape[0].upper()}"

class Qwirkle:
    def __init__(self, num_players=2):
        self.num_players = num_players
        self.reset()
    
    def reset(self):
        # Create tile bag (3 of each combination)
        self.bag = []
        for color in COLORS:
            for shape in SHAPES:
                for _ in range(3):
                    self.bag.append(Tile(color, shape))
        random.shuffle(self.bag)
        
        # Initialize board as dict {(row, col): Tile}
        self.board = {}
        
        # Initialize player hands
        self.hands = [self._draw_tiles(6) for _ in range(self.num_players)]
        
        # Game state
        self.current_player = 0
        self.scores = [0] * self.num_players
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.passes = 0
        
        return self.get_state()
    
    def _draw_tiles(self, count):
        drawn = []
        for _ in range(min(count, len(self.bag))):
            if self.bag:
                drawn.append(self.bag.pop())
        return drawn
    
    def get_state(self):
        # Simplified state representation
        state_data = {
            'board_size': len(self.board),
            'hand_size': len(self.hands[self.current_player]),
            'tiles_left': len(self.bag),
            'score_diff': self.scores[self.current_player] - max([s for i, s in enumerate(self.scores) if i != self.current_player], default=0)
        }
        return tuple(sorted(state_data.items()))
    
    def get_available_actions(self):
        """Return list of legal moves"""
        actions = []
        hand = self.hands[self.current_player]
        
        if not self.board:
            # First move: can place any tiles that match
            for i in range(len(hand)):
                actions.append(('place', [(0, 0, i)]))  # Single tile at origin
                
                # Try placing multiple matching tiles
                for j in range(i + 1, len(hand)):
                    if self._tiles_compatible([hand[i], hand[j]]):
                        actions.append(('place', [(0, 0, i), (0, 1, j)]))
        else:
            # Find all valid placements
            empty_neighbors = self._get_empty_neighbors()
            
            # Single tile placements
            for pos in empty_neighbors:
                for idx, tile in enumerate(hand):
                    if self._is_valid_placement(pos, tile):
                        actions.append(('place', [(pos[0], pos[1], idx)]))
            
            # Multi-tile placements (simplified - just try 2-tile combos)
            if len(hand) >= 2:
                for pos1 in list(empty_neighbors)[:10]:  # Limit search
                    for idx1, tile1 in enumerate(hand[:4]):
                        if self._is_valid_placement(pos1, tile1):
                            # Try placing second tile in adjacent positions
                            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                pos2 = (pos1[0] + dr, pos1[1] + dc)
                                if pos2 not in self.board:
                                    for idx2, tile2 in enumerate(hand[:4]):
                                        if idx2 != idx1 and self._tiles_compatible([tile1, tile2]):
                                            actions.append(('place', [(pos1[0], pos1[1], idx1), (pos2[0], pos2[1], idx2)]))
        
        # Can always trade tiles
        actions.append(('trade', []))
        
        return actions[:100]  # Limit action space
    
    def _get_empty_neighbors(self):
        """Get all empty positions adjacent to placed tiles"""
        neighbors = set()
        for (r, c) in self.board:
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in self.board:
                    neighbors.add((nr, nc))
        return neighbors
    
    def _tiles_compatible(self, tiles):
        """Check if tiles can be in same line"""
        if len(tiles) <= 1:
            return True
        
        # Check if all same color or all same shape
        colors = set(t.color for t in tiles)
        shapes = set(t.shape for t in tiles)
        
        if len(colors) == 1 and len(shapes) == len(tiles):
            return True  # Same color, different shapes
        if len(shapes) == 1 and len(colors) == len(tiles):
            return True  # Same shape, different colors
        return False
    
    def _is_valid_placement(self, pos, tile):
        """Check if tile can be placed at position"""
        r, c = pos
        
        if (r, c) in self.board:
            return False
        
        # Check adjacent tiles
        adjacent_tiles = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            adj_pos = (r + dr, c + dc)
            if adj_pos in self.board:
                adjacent_tiles.append(self.board[adj_pos])
        
        if not adjacent_tiles:
            return len(self.board) == 0  # Only valid if board is empty
        
        # Check horizontal line
        h_tiles = [tile]
        for dc in [-1, 1]:
            cc = c + dc
            while (r, cc) in self.board:
                h_tiles.append(self.board[(r, cc)])
                cc += dc
        
        if len(h_tiles) > 1 and not self._tiles_compatible(h_tiles):
            return False
        
        # Check vertical line
        v_tiles = [tile]
        for dr in [-1, 1]:
            rr = r + dr
            while (rr, c) in self.board:
                v_tiles.append(self.board[(rr, c)])
                rr += dr
        
        if len(v_tiles) > 1 and not self._tiles_compatible(v_tiles):
            return False
        
        # Check line length constraints
        if len(h_tiles) > 6 or len(v_tiles) > 6:
            return False
        
        return True
    
    def make_move(self, action):
        """Execute a move and return (state, reward, done)"""
        if self.game_over:
            return self.get_state(), 0, True
        
        action_type, data = action
        reward = 0
        
        if action_type == 'place':
            # Place tiles on board
            placed_tiles = []
            for r, c, hand_idx in data:
                tile = self.hands[self.current_player][hand_idx]
                self.board[(r, c)] = tile
                placed_tiles.append((r, c, hand_idx))
            
            # Calculate score
            reward = self._calculate_placement_score(placed_tiles)
            self.scores[self.current_player] += reward
            
            # Remove tiles from hand (in reverse order to maintain indices)
            for _, _, hand_idx in sorted(placed_tiles, key=lambda x: x[2], reverse=True):
                self.hands[self.current_player].pop(hand_idx)
            
            # Draw new tiles
            new_tiles = self._draw_tiles(len(placed_tiles))
            self.hands[self.current_player].extend(new_tiles)
            
            self.passes = 0
            
        elif action_type == 'trade':
            # Trade all tiles
            traded = self.hands[self.current_player][:]
            self.hands[self.current_player] = []
            self.bag.extend(traded)
            random.shuffle(self.bag)
            self.hands[self.current_player] = self._draw_tiles(len(traded))
            reward = -2  # Small penalty for trading
            self.passes += 1
        
        self.move_history.append((action, self.current_player))
        
        # Check end game
        if len(self.hands[self.current_player]) == 0 and len(self.bag) == 0:
            self.game_over = True
            self.scores[self.current_player] += 6  # Bonus for finishing
            self.winner = self.scores.index(max(self.scores))
            return self.get_state(), reward + 50, True
        
        if self.passes >= self.num_players * 2:
            self.game_over = True
            self.winner = self.scores.index(max(self.scores))
            return self.get_state(), reward, True
        
        # Switch player
        self.current_player = (self.current_player + 1) % self.num_players
        
        return self.get_state(), reward, False
    
    def _calculate_placement_score(self, placed_tiles):
        """Calculate score for placed tiles"""
        score = 0
        scored_positions = set()
        
        for r, c, _ in placed_tiles:
            # Score horizontal line
            h_line = set()
            for dc in range(-6, 7):
                if (r, c + dc) in self.board:
                    h_line.add((r, c + dc))
                else:
                    if dc < 0:
                        continue
                    break
            
            if len(h_line) > 1:
                line_score = len(h_line)
                if len(h_line) == 6:
                    line_score += 6  # Qwirkle bonus
                if not h_line.issubset(scored_positions):
                    score += line_score
                    scored_positions.update(h_line)
            
            # Score vertical line
            v_line = set()
            for dr in range(-6, 7):
                if (r + dr, c) in self.board:
                    v_line.add((r + dr, c))
                else:
                    if dr < 0:
                        continue
                    break
            
            if len(v_line) > 1:
                line_score = len(v_line)
                if len(v_line) == 6:
                    line_score += 6  # Qwirkle bonus
                if not v_line.issubset(scored_positions):
                    score += line_score
                    scored_positions.update(v_line)
        
        if score == 0 and len(placed_tiles) > 0:
            score = len(placed_tiles)
        
        return score
    
    def evaluate_position(self, player):
        """Advanced position evaluation"""
        if self.winner == player:
            return 100000
        if self.winner is not None and self.winner != player:
            return -100000
        
        score = 0
        
        # 1. Score advantage
        my_score = self.scores[player]
        opp_scores = [self.scores[i] for i in range(self.num_players) if i != player]
        score += (my_score - max(opp_scores, default=0)) * 100
        
        # 2. Hand quality
        hand = self.hands[player]
        score += len(hand) * 10
        
        # 3. Potential Qwirkle opportunities
        for pos in self._get_empty_neighbors() if self.board else [(0, 0)]:
            for tile in hand[:3]:  # Sample hand
                if self._is_valid_placement(pos, tile):
                    # Check if placement leads to near-Qwirkle
                    r, c = pos
                    for dr, dc in [(0, 1), (1, 0)]:
                        line_len = 1
                        for d in [-1, 1]:
                            rr, cc = r, c
                            while True:
                                rr, cc = rr + d * dr, cc + d * dc
                                if (rr, cc) in self.board:
                                    line_len += 1
                                else:
                                    break
                        if line_len >= 4:
                            score += (line_len * 20)
        
        # 4. Tile diversity
        colors_in_hand = len(set(t.color for t in hand))
        shapes_in_hand = len(set(t.shape for t in hand))
        score += (colors_in_hand + shapes_in_hand) * 5
        
        return score

# ============================================================================
# MCTS Node
# ============================================================================

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None
    
    def uct_value(self, exploration=1.41):
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
    
    def best_child(self, exploration=1.41):
        return max(self.children, key=lambda c: c.uct_value(exploration))
    
    def most_visited_child(self):
        return max(self.children, key=lambda c: c.visits)

# ============================================================================
# Strategic RL Agent with MCTS
# ============================================================================

class StrategicQwirkleAgent:
    def __init__(self, player_id, lr=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.05):
        self.player_id = player_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_table = {}
        self.experience_replay = deque(maxlen=10000)
        self.mcts_simulations = 50  # MCTS iterations per move
        
        # Stats
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_score = 0
        self.games_played = 0
    
    def get_q_value(self, state, action):
        action_key = str(action)
        return self.q_table.get((state, action_key), 0.0)
    
    def choose_action(self, env, training=True):
        available_actions = env.get_available_actions()
        if not available_actions:
            return None
        
        # Level 1: Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # Level 2: MCTS strategic planning
        best_action = self._mcts_search(env, available_actions)
        
        return best_action
    
    def _mcts_search(self, env, actions):
        """Monte Carlo Tree Search"""
        root = MCTSNode(env.get_state())
        root.untried_actions = actions[:]
        
        for _ in range(self.mcts_simulations):
            node = root
            sim_env = self._copy_env(env)
            
            # Selection
            while node.untried_actions == [] and node.children:
                node = node.best_child()
                if node.action:
                    sim_env.make_move(node.action)
            
            # Expansion
            if node.untried_actions:
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                sim_env.make_move(action)
                child = MCTSNode(sim_env.get_state(), parent=node, action=action)
                node.children.append(child)
                node = child
            
            # Simulation
            reward = self._simulate_random_playout(sim_env)
            
            # Backpropagation
            while node:
                node.visits += 1
                node.value += reward
                node = node.parent
        
        # Return most visited action
        if root.children:
            best_child = root.most_visited_child()
            return best_child.action
        
        # Fallback: best Q-value
        best_score = -float('inf')
        best_action = actions[0]
        for action in actions:
            q_value = self.get_q_value(env.get_state(), action)
            if q_value > best_score:
                best_score = q_value
                best_action = action
        
        return best_action
    
    def _simulate_random_playout(self, env):
        """Simulate game to end with random moves"""
        discount = 1.0
        total_reward = 0
        max_steps = 30
        steps = 0
        
        while not env.game_over and steps < max_steps:
            actions = env.get_available_actions()
            if not actions:
                break
            action = random.choice(actions)
            _, reward, done = env.make_move(action)
            total_reward += reward * discount
            discount *= 0.95
            steps += 1
        
        # Add final score differential
        if env.game_over and env.winner == self.player_id:
            total_reward += 100
        elif env.game_over:
            total_reward -= 50
        
        return total_reward
    
    def _copy_env(self, env):
        """Create a lightweight copy of environment"""
        new_env = Qwirkle(env.num_players)
        new_env.board = env.board.copy()
        new_env.hands = [hand[:] for hand in env.hands]
        new_env.bag = env.bag[:]
        new_env.current_player = env.current_player
        new_env.scores = env.scores[:]
        new_env.game_over = env.game_over
        new_env.winner = env.winner
        new_env.passes = env.passes
        return new_env
    
    def update_q_value(self, state, action, reward, next_state, next_available_actions):
        action_key = str(action)
        current_q = self.get_q_value(state, action)
        
        if next_available_actions:
            max_next_q = max([self.get_q_value(next_state, a) for a in next_available_actions[:10]], default=0)
        else:
            max_next_q = 0
        
        td_error = reward + self.gamma * max_next_q - current_q
        new_q = current_q + self.lr * td_error
        self.q_table[(state, action_key)] = new_q
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_stats(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_score = 0
        self.games_played = 0

# ============================================================================
# Training System
# ============================================================================

def play_game(env, agent1, agent2, training=True):
    """Play one complete game"""
    env.reset()
    game_history = []
    
    agents = [agent1, agent2]
    
    max_moves = 200
    move_count = 0
    
    while not env.game_over and move_count < max_moves:
        current_player = env.current_player
        current_agent = agents[current_player]
        
        state = env.get_state()
        action = current_agent.choose_action(env, training)
        
        if action is None:
            break
        
        game_history.append((state, action, current_player))
        next_state, reward, done = env.make_move(action)
        
        if training:
            next_actions = env.get_available_actions()
            current_agent.update_q_value(state, action, reward, next_state, next_actions)
        
        move_count += 1
        
        if done:
            # Update stats
            for i, agent in enumerate(agents):
                agent.games_played += 1
                agent.total_score += env.scores[i]
                
                if env.winner == i:
                    agent.wins += 1
                    if training:
                        _update_from_outcome(agent, game_history, i, 100)
                elif env.winner is not None:
                    agent.losses += 1
                    if training:
                        _update_from_outcome(agent, game_history, i, -50)
                else:
                    agent.draws += 1
    
    return env.winner, env.scores

def _update_from_outcome(agent, history, player_id, final_reward):
    """Update agent's strategy based on game outcome"""
    agent_moves = [(s, a) for s, a, p in history if p == player_id]
    
    for i in range(len(agent_moves) - 1, -1, -1):
        state, action = agent_moves[i]
        discount_factor = agent.gamma ** (len(agent_moves) - 1 - i)
        adjusted_reward = final_reward * discount_factor
        
        action_key = str(action)
        current_q = agent.get_q_value(state, action)
        new_q = current_q + agent.lr * (adjusted_reward - current_q)
        agent.q_table[(state, action_key)] = new_q

# ============================================================================
# Visualization
# ============================================================================

def visualize_board(env, title="Qwirkle Board"):
    """Create matplotlib figure of the Qwirkle board"""
    if not env.board:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'Board is Empty', ha='center', va='center', fontsize=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')
        return fig
    
    # Get board bounds
    positions = list(env.board.keys())
    min_r = min(r for r, c in positions)
    max_r = max(r for r, c in positions)
    min_c = min(c for r, c in positions)
    max_c = max(c for r, c in positions)
    
    # Add padding
    min_r -= 1
    max_r += 1
    min_c -= 1
    max_c += 1
    
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    
    fig, ax = plt.subplots(figsize=(min(16, width * 0.8), min(12, height * 0.8)))
    
    # Draw grid
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            rect = Rectangle((c, -r), 1, 1, facecolor='#2c3e50', 
                           edgecolor='#34495e', linewidth=1)
            ax.add_patch(rect)
            
            if (r, c) in env.board:
                tile = env.board[(r, c)]
                # Draw colored square
                inner_rect = Rectangle((c + 0.15, -r + 0.15), 0.7, 0.7,
                                      facecolor=COLOR_CODES[tile.color],
                                      edgecolor='white', linewidth=2)
                ax.add_patch(inner_rect)
                
                # Draw shape symbol
                symbol = SHAPE_SYMBOLS[tile.shape]
                ax.text(c + 0.5, -r + 0.5, symbol, ha='center', va='center',
                       fontsize=24, color='white', weight='bold')
    
    ax.set_xlim(min_c - 0.5, max_c + 1.5)
    ax.set_ylim(-max_r - 1.5, -min_r + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    return fig

# ============================================================================
# Save/Load Functions
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def serialize_q_table(q_table):
    serialized_q = {}
    for (state, action_key), value in q_table.items():
        state_str = json.dumps(list(state))
        key_str = f"{state_str}|{action_key}"
        serialized_q[key_str] = float(value)
    return serialized_q

def deserialize_q_table(serialized_q):
    deserialized_q = {}
    for k_str, value in serialized_q.items():
        try:
            state_str, action_key = k_str.split('|', 1)
            state_list = json.loads(state_str)
            state_tuple = tuple(state_list)
            deserialized_q[(state_tuple, action_key)] = value
        except:
            continue
    return deserialized_q

def create_agents_zip(agent1, agent2, config):
    agent1_data = {
        "q_table": serialize_q_table(agent1.q_table),
        "epsilon": float(agent1.epsilon),
        "wins": int(agent1.wins),
        "losses": int(agent1.losses),
        "total_score": int(agent1.total_score),
        "games_played": int(agent1.games_played)
    }
    
    agent2_data = {
        "q_table": serialize_q_table(agent2.q_table),
        "epsilon": float(agent2.epsilon),
        "wins": int(agent2.wins),
        "losses": int(agent2.losses),
        "total_score": int(agent2.total_score),
        "games_played": int(agent2.games_played)
    }
    
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("agent1.json", json.dumps(agent1_data, cls=NumpyEncoder, indent=2))
        zf.writestr("agent2.json", json.dumps(agent2_data, cls=NumpyEncoder, indent=2))
        zf.writestr("config.json", json.dumps(config, cls=NumpyEncoder, indent=2))
    
    buffer.seek(0)
    return buffer

def load_agents_from_zip(uploaded_file):
    try:
        with zipfile.ZipFile(uploaded_file, "r") as zf:
            agent1_state = json.loads(zf.read("agent1.json"))
            agent2_state = json.loads(zf.read("agent2.json"))
            config = json.loads(zf.read("config.json"))
            
            agent1 = StrategicQwirkleAgent(0, config.get('lr1', 0.1), config.get('gamma1', 0.95))
            agent1.q_table = deserialize_q_table(agent1_state['q_table'])
            agent1.epsilon = agent1_state.get('epsilon', 0.0)
            agent1.wins = agent1_state.get('wins', 0)
            agent1.losses = agent1_state.get('losses', 0)
            agent1.total_score = agent1_state.get('total_score', 0)
            agent1.games_played = agent1_state.get('games_played', 0)
            
            agent2 = StrategicQwirkleAgent(1, config.get('lr2', 0.1), config.get('gamma2', 0.95))
            agent2.q_table = deserialize_q_table(agent2_state['q_table'])
            agent2.epsilon = agent2_state.get('epsilon', 0.0)
            agent2.wins = agent2_state.get('wins', 0)
            agent2.losses = agent2_state.get('losses', 0)
            agent2.total_score = agent2_state.get('total_score', 0)
            agent2.games_played = agent2_state.get('games_played', 0)
            
            return agent1, agent2, config
    except Exception as e:
        st.error(f"Error loading brain file: {str(e)}")
        return None, None, None

# ============================================================================
# Streamlit UI
# ============================================================================

st.sidebar.header("⚙️ Simulation Controls")

with st.sidebar.expander("1. Agent 1 Parameters", expanded=True):
    lr1 = st.slider("Learning Rate α₁", 0.01, 0.5, 0.1, 0.01)
    gamma1 = st.slider("Discount Factor γ₁", 0.8, 0.99, 0.95, 0.01)
    epsilon_decay1 = st.slider("Epsilon Decay₁", 0.99, 0.9999, 0.995, 0.0001, format="%.4f")
    mcts_sims1 = st.slider("MCTS Simulations₁", 10, 200, 50, 10)

with st.sidebar.expander("2. Agent 2 Parameters", expanded=True):
    lr2 = st.slider("Learning Rate α₂", 0.01, 0.5, 0.1, 0.01)
    gamma2 = st.slider("Discount Factor γ₂", 0.8, 0.99, 0.95, 0.01)
    epsilon_decay2 = st.slider("Epsilon Decay₂", 0.99, 0.9999, 0.995, 0.0001, format="%.4f")
    mcts_sims2 = st.slider("MCTS Simulations₂", 10, 200, 50, 10)

with st.sidebar.expander("3. Training Configuration", expanded=True):
    episodes = st.number_input("Training Episodes", 10, 10000, 100, 10)
    update_freq = st.number_input("Update Dashboard Every N Games", 5, 100, 10, 5)

with st.sidebar.expander("4. Brain Storage 💾", expanded=False):
    if 'agent1' in st.session_state and st.session_state.agent1 is not None:
        brain_size = len(st.session_state.agent1.q_table) + len(st.session_state.agent2.q_table)
        
        if brain_size > 0:
            st.success(f"🧠 Brain Scan: {brain_size} memories found.")
            
            config = {
                "lr1": lr1, "gamma1": gamma1, "epsilon_decay1": epsilon_decay1, "mcts_sims1": mcts_sims1,
                "lr2": lr2, "gamma2": gamma2, "epsilon_decay2": epsilon_decay2, "mcts_sims2": mcts_sims2,
                "training_history": st.session_state.get('training_history', None)
            }
            
            zip_buffer = create_agents_zip(st.session_state.agent1, st.session_state.agent2, config)
            
            st.download_button(
                label="💾 Download Trained Brains",
                data=zip_buffer,
                file_name="qwirkle_brains.zip",
                mime="application/zip",
                use_container_width=True
            )
        else:
            st.warning("⚠️ Brains are empty! Train the agents before downloading.")
    else:
        st.warning("Initialize agents first.")
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload Brain Snapshot (.zip)", type="zip")
    if uploaded_file is not None:
        if st.button("📂 Load Session", use_container_width=True):
            with st.spinner("Restoring neural pathways..."):
                a1, a2, cfg = load_agents_from_zip(uploaded_file)
                if a1:
                    st.session_state.agent1 = a1
                    st.session_state.agent2 = a2
                    
                    if cfg.get("training_history"):
                        st.session_state.training_history = cfg["training_history"]
                    
                    st.toast("Brains Restored Successfully!", icon="♾️")
                    import time
                    time.sleep(1)
                    st.rerun()

train_button = st.sidebar.button("🚀 Begin Training Epochs", 
                                 use_container_width=True, type="primary")

if st.sidebar.button("🧹 Clear All & Reset", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.cache_data.clear()
    st.toast("Simulation Arena Reset!", icon="🧹")
    st.rerun()

# ============================================================================
# Initialize Environment and Agents
# ============================================================================

if 'env' not in st.session_state:
    st.session_state.env = Qwirkle()

if 'agent1' not in st.session_state:
    st.session_state.agent1 = StrategicQwirkleAgent(0, lr1, gamma1, epsilon_decay=epsilon_decay1)
    st.session_state.agent1.mcts_simulations = mcts_sims1
    st.session_state.agent2 = StrategicQwirkleAgent(1, lr2, gamma2, epsilon_decay=epsilon_decay2)
    st.session_state.agent2.mcts_simulations = mcts_sims2

agent1 = st.session_state.agent1
agent2 = st.session_state.agent2
env = st.session_state.env

agent1.mcts_simulations = mcts_sims1
agent2.mcts_simulations = mcts_sims2

# ============================================================================
# Display Current Stats
# ============================================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎨 Agent 1", f"Q-States: {len(agent1.q_table)}", f"ε={agent1.epsilon:.4f}")
    st.metric("Wins", agent1.wins, delta_color="normal")
    avg_score = agent1.total_score / max(agent1.games_played, 1)
    st.caption(f"Avg Score: {avg_score:.1f} | MCTS Sims: {agent1.mcts_simulations}")

with col2:
    st.metric("🎨 Agent 2", f"Q-States: {len(agent2.q_table)}", f"ε={agent2.epsilon:.4f}")
    st.metric("Wins", agent2.wins, delta_color="normal")
    avg_score = agent2.total_score / max(agent2.games_played, 1)
    st.caption(f"Avg Score: {avg_score:.1f} | MCTS Sims: {agent2.mcts_simulations}")

with col3:
    total_games = agent1.games_played
    st.metric("Total Games", total_games)
    st.metric("Draws", agent1.draws, delta_color="off")

st.markdown("---")

# ============================================================================
# Training Section
# ============================================================================

if train_button:
    st.subheader("🎯 Training Epochs in Progress...")
    
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    agent1.reset_stats()
    agent2.reset_stats()
    
    history = {
        'agent1_wins': [],
        'agent2_wins': [],
        'draws': [],
        'agent1_epsilon': [],
        'agent2_epsilon': [],
        'agent1_q_size': [],
        'agent2_q_size': [],
        'agent1_avg_score': [],
        'agent2_avg_score': [],
        'episode': []
    }
    
    for episode in range(1, episodes + 1):
        winner, scores = play_game(env, agent1, agent2, training=True)
        
        agent1.decay_epsilon()
        agent2.decay_epsilon()
        
        if episode % update_freq == 0:
            history['agent1_wins'].append(agent1.wins)
            history['agent2_wins'].append(agent2.wins)
            history['draws'].append(agent1.draws)
            history['agent1_epsilon'].append(agent1.epsilon)
            history['agent2_epsilon'].append(agent2.epsilon)
            history['agent1_q_size'].append(len(agent1.q_table))
            history['agent2_q_size'].append(len(agent2.q_table))
            
            avg1 = agent1.total_score / max(agent1.games_played, 1)
            avg2 = agent2.total_score / max(agent2.games_played, 1)
            history['agent1_avg_score'].append(avg1)
            history['agent2_avg_score'].append(avg2)
            history['episode'].append(episode)
            
            progress = episode / episodes
            progress_bar.progress(progress)
            
            status_table = f"""
            | Metric          | Agent 1 | Agent 2 |
            |:----------------|:-------:|:-------:|
            | **Wins**        | {agent1.wins} | {agent2.wins} |
            | **Epsilon (ε)** | {agent1.epsilon:.4f} | {agent2.epsilon:.4f} |
            | **Q-States**    | {len(agent1.q_table):,} | {len(agent2.q_table):,} |
            | **Avg Score**   | {avg1:.1f} | {avg2:.1f} |
            
            ---
            **Game {episode}/{episodes}** ({progress*100:.1f}%) | **Total Draws:** {agent1.draws}
            """
            status_container.markdown(status_table)
    
    progress_bar.progress(1.0)
    
    st.session_state.training_history = history
    st.session_state.agent1 = agent1
    st.session_state.agent2 = agent2
    
    st.toast("Training Complete! Refreshing UI...", icon="🎉")
    
    import time
    time.sleep(1)
    st.rerun()

# ============================================================================
# Display Training Charts
# ============================================================================

if 'training_history' in st.session_state and st.session_state.training_history:
    st.subheader("📈 Training Performance Analysis")
    history = st.session_state.training_history
    
    df = pd.DataFrame(history)
    
    if 'episode' not in df.columns:
        df['episode'] = range(1, len(df) + 1)
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.write("#### Win/Loss/Draw Count Over Time")
        chart_data = df[['episode', 'agent1_wins', 'agent2_wins', 'draws']].set_index('episode')
        st.line_chart(chart_data)
    
    with chart_col2:
        st.write("#### Average Score Per Game")
        if 'agent1_avg_score' in df.columns:
            chart_data = df[['episode', 'agent1_avg_score', 'agent2_avg_score']].set_index('episode')
            st.line_chart(chart_data)
    
    st.write("#### Epsilon Decay (Exploration Rate)")
    epsilon_data = df[['episode', 'agent1_epsilon', 'agent2_epsilon']].set_index('episode')
    st.line_chart(epsilon_data)
    
    st.write("#### Q-Table Size (Learned States)")
    q_chart_data = df[['episode', 'agent1_q_size', 'agent2_q_size']].set_index('episode')
    st.line_chart(q_chart_data)

# ============================================================================
# Final Battle Visualization
# ============================================================================

if 'agent1' in st.session_state and st.session_state.agent1.q_table:
    st.subheader("⚔️ Final Battle: Trained Agents")
    st.info("Watch the fully trained agents play one final game against each other (no exploration).")
    
    if st.button("🎮 Watch Them Battle!", use_container_width=True):
        sim_env = Qwirkle()
        board_placeholder = st.empty()
        score_placeholder = st.empty()
        
        agents = [agent1, agent2]
        
        move_count = 0
        max_moves = 200
        
        with st.spinner("Agents are battling..."):
            while not sim_env.game_over and move_count < max_moves:
                current_player = sim_env.current_player
                action = agents[current_player].choose_action(sim_env, training=False)
                if action is None:
                    break
                sim_env.make_move(action)
                
                player_name = f"Agent {current_player + 1}"
                fig = visualize_board(sim_env, f"{player_name}'s move | Move {move_count + 1}")
                board_placeholder.pyplot(fig)
                plt.close(fig)
                
                score_placeholder.markdown(f"**Scores:** Agent 1: {sim_env.scores[0]} | Agent 2: {sim_env.scores[1]}")
                
                import time
                time.sleep(0.5)
                move_count += 1
        
        if sim_env.winner == 0:
            st.success(f"🏆 Agent 1 wins! Final Score: {sim_env.scores[0]} - {sim_env.scores[1]}")
        elif sim_env.winner == 1:
            st.error(f"🏆 Agent 2 wins! Final Score: {sim_env.scores[0]} - {sim_env.scores[1]}")
        else:
            st.warning(f"🤝 Draw! Final Score: {sim_env.scores[0]} - {sim_env.scores[1]}")
else:
    st.info("Train or load agents to see the Final Battle option.")

st.markdown("---")
st.caption("Strategic RL Qwirkle Arena | Powered by MCTS + Q-Learning 🎨")
