import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy
import random

# ============================================================================
# Page Config
# ============================================================================
st.set_page_config(
    page_title="Quantum Match",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚛"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Inter:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.title-block {
    background: linear-gradient(135deg, #000814 0%, #001d3d 50%, #000814 100%);
    border: 1px solid #0077b6;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    box-shadow: 0 0 50px rgba(0,119,182,0.2);
}
.title-block h1 { color: #48cae4; font-size: 2.4rem; margin: 0; letter-spacing: -1px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.title-block p  { color: #90e0ef; font-size: 0.93rem; margin: 6px 0 0; opacity: 0.85; }

.rule-card {
    background: #000d1a;
    border: 1px solid #0d2137;
    border-left: 3px solid #0077b6;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    color: #caf0f8;
    font-size: 0.86rem;
    line-height: 1.6;
}
.stat-card {
    background: #000d1a;
    border: 1px solid #0d2137;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
}
.stat-card .val { font-size: 2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.stat-card .lbl { font-size: 0.78rem; color: #4a7fa5; margin-top: 2px; }

.win-banner {
    background: linear-gradient(90deg, #023e8a, #0077b6);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
    color: #caf0f8;
    margin: 16px 0;
    box-shadow: 0 0 30px rgba(0,119,182,0.4);
    font-family: 'JetBrains Mono', monospace;
}
.turn-indicator {
    background: #000d1a;
    border: 1px solid #0d2137;
    border-radius: 10px;
    padding: 12px 18px;
    margin: 10px 0;
    font-size: 1rem;
    color: #90e0ef;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
}
.action-section {
    background: #000d1a;
    border: 1px solid #0d2137;
    border-radius: 12px;
    padding: 16px;
    margin: 10px 0;
}
.pool-display {
    font-family: 'JetBrains Mono', monospace;
    color: #48cae4;
    font-size: 1.1rem;
    background: #000d1a;
    border: 1px solid #023e8a;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-block">
  <h1>⚛ Quantum Match</h1>
  <p>Hidden-information deduction · 2×2 grid · Shared token pool · Flip or Place · 81 states</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Game Constants
# ============================================================================
EMPTY  = 0
WHITE  = 1   # Player 1
BLACK  = 2   # Player 2

GRID_SIZE = 4   # 2x2 = 4 squares
POOL_SIZE = 4   # shared pool starts with 4 tokens

# ============================================================================
# Game State
# ============================================================================
def initial_state():
    return {
        "grid": [EMPTY] * 4,     # 4 squares: index 0-3 (row-major)
        "pool": 4,               # tokens remaining in shared pool
        "turn": WHITE,
        "game_over": False,
        "winner": None,
        "turn_count": 0,
        "last_action": None,
        "history": []
    }

def check_winner(grid):
    """Win: all 4 squares show your color"""
    if all(v == WHITE for v in grid):
        return WHITE
    if all(v == BLACK for v in grid):
        return BLACK
    return None

def is_terminal(state):
    if state["game_over"]:
        return True
    # If no pool tokens AND no pieces to flip on the board, game ends in draw
    return False

def get_valid_actions(state):
    """
    Actions:
      ("place", idx)  — place token from pool on empty square with current color up
      ("flip",  idx)  — flip existing token at idx to current color
    """
    actions = []
    player = state["turn"]
    opponent = 3 - player

    # Can place if pool > 0 and square is empty
    if state["pool"] > 0:
        for i, v in enumerate(state["grid"]):
            if v == EMPTY:
                actions.append(("place", i))

    # Can flip any opponent's square to own color
    for i, v in enumerate(state["grid"]):
        if v == opponent:
            actions.append(("flip", i))

    return actions

def apply_action(state, action):
    s = deepcopy(state)
    atype, idx = action
    player = s["turn"]

    if atype == "place":
        s["grid"][idx] = player
        s["pool"] -= 1
        s["last_action"] = f"{'White' if player==WHITE else 'Black'} placed at {_pos_name(idx)}"
    elif atype == "flip":
        s["grid"][idx] = player
        s["last_action"] = f"{'White' if player==WHITE else 'Black'} flipped {_pos_name(idx)}"

    s["turn_count"] += 1
    s["history"].append((atype, idx, player))

    winner = check_winner(s["grid"])
    if winner:
        s["game_over"] = True
        s["winner"] = winner
    elif not get_valid_actions(s):
        # No moves available → current player loses
        s["game_over"] = True
        s["winner"] = 3 - player

    s["turn"] = 3 - player
    return s

def _pos_name(idx):
    names = ["TL", "TR", "BL", "BR"]
    return names[idx]

# ============================================================================
# AI — Minimax
# ============================================================================
def evaluate(state, ai_player):
    grid = state["grid"]
    opp  = 3 - ai_player

    if state["winner"] == ai_player:
        return 10000
    if state["winner"] == opp:
        return -10000

    # Count owned squares
    mine = grid.count(ai_player)
    theirs = grid.count(opp)

    # Strategic: corners matter, center adjacency
    corner_bonus = 0
    for i in [0, 1, 2, 3]:
        if grid[i] == ai_player:
            corner_bonus += 5

    score = (mine - theirs) * 20 + corner_bonus

    # Threat: if 3 in a row owned = urgent
    if mine == 3:
        score += 50
    if theirs == 3:
        score -= 80  # Block more important

    # Pool advantage
    if state["pool"] == 0:
        # No new placements, only flips matter
        score += mine * 10

    return score

def minimax_qm(state, depth, alpha, beta, maximizing, ai_player, visited=None):
    if visited is None:
        visited = set()

    key = (tuple(state["grid"]), state["pool"], state["turn"])
    if key in visited and depth < 3:
        return 0, None
    visited.add(key)

    if state["game_over"] or depth == 0:
        return evaluate(state, ai_player), None

    actions = get_valid_actions(state)
    if not actions:
        return evaluate(state, ai_player), None

    best_action = None

    if maximizing:
        best_val = -float('inf')
        for a in actions:
            ns = apply_action(state, a)
            val, _ = minimax_qm(ns, depth - 1, alpha, beta, False, ai_player, visited.copy())
            if val > best_val:
                best_val, best_action = val, a
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return best_val, best_action
    else:
        best_val = float('inf')
        for a in actions:
            ns = apply_action(state, a)
            val, _ = minimax_qm(ns, depth - 1, alpha, beta, True, ai_player, visited.copy())
            if val < best_val:
                best_val, best_action = val, a
            beta = min(beta, val)
            if beta <= alpha:
                break
        return best_val, best_action

def ai_choose(state, ai_player, difficulty):
    depths = {"Easy": 2, "Medium": 4, "Hard": 7}
    d = depths[difficulty]
    if difficulty == "Easy" and random.random() < 0.4:
        actions = get_valid_actions(state)
        return random.choice(actions) if actions else None
    _, action = minimax_qm(state, d, -float('inf'), float('inf'), True, ai_player)
    return action

# ============================================================================
# Visualization
# ============================================================================
def draw_board(state):
    grid = state["grid"]
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("#000814")
    ax.set_facecolor("#000814")

    # Color maps
    colors = {
        EMPTY: "#001d3d",
        WHITE: "#caf0f8",
        BLACK: "#03045e"
    }
    edge_colors = {
        EMPTY: "#0077b6",
        WHITE: "#48cae4",
        BLACK: "#023e8a"
    }
    labels = {EMPTY: "?", WHITE: "W", BLACK: "B"}
    label_colors = {EMPTY: "#0077b6", WHITE: "#000814", BLACK: "#90e0ef"}

    positions = [(0, 1), (1, 1), (0, 0), (1, 0)]  # TL, TR, BL, BR
    sq = 0.9

    for idx, (col_i, row_i) in enumerate(positions):
        piece = grid[idx]
        x, y = col_i * 1.2, row_i * 1.2
        rect = plt.Rectangle((x, y), sq, sq,
                              facecolor=colors[piece],
                              edgecolor=edge_colors[piece],
                              linewidth=3, zorder=2)
        ax.add_patch(rect)

        # Index label
        ax.text(x + sq/2, y + sq/2 + 0.1,
                labels[piece],
                ha="center", va="center",
                fontsize=28, fontweight="bold",
                color=label_colors[piece], zorder=3,
                fontfamily="monospace")
        ax.text(x + sq/2, y + 0.12,
                _pos_name(idx),
                ha="center", va="center",
                fontsize=9, color="#4a7fa5", zorder=3)

    # Pool display
    pool_str = "◈ " * state["pool"] + "○ " * (POOL_SIZE - state["pool"])
    ax.text(1.05, -0.3, f"Pool: {state['pool']}/4  {pool_str}",
            ha="center", va="center",
            fontsize=11, color="#48cae4", fontfamily="monospace")

    ax.set_xlim(-0.2, 2.3)
    ax.set_ylim(-0.55, 2.3)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    return fig

def draw_probability_map(state, ai_player):
    """Show which squares AI considers most valuable"""
    grid = state["grid"]
    values = []
    for i in range(4):
        ns = deepcopy(state)
        ns["grid"][i] = ai_player
        v = evaluate(ns, ai_player)
        values.append(v)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.patch.set_facecolor("#000814")
    ax.set_facecolor("#000814")

    positions = [(0, 1), (1, 1), (0, 0), (1, 0)]
    sq = 0.88
    vmin, vmax = min(values), max(values)
    vrange = max(vmax - vmin, 1)

    cmap = plt.cm.cool
    for idx, (col_i, row_i) in enumerate(positions):
        x, y = col_i * 1.1, row_i * 1.1
        intensity = (values[idx] - vmin) / vrange
        facecolor = cmap(intensity)
        rect = plt.Rectangle((x, y), sq, sq,
                              facecolor=facecolor,
                              edgecolor="#023e8a", linewidth=2, zorder=2)
        ax.add_patch(rect)
        ax.text(x + sq/2, y + sq/2, f"{values[idx]:+.0f}",
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white", zorder=3, fontfamily="monospace")

    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.1, 2.1)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    return fig

# ============================================================================
# Session State
# ============================================================================
def reset_game():
    st.session_state.game_state = initial_state()
    if "scores" not in st.session_state:
        st.session_state.scores = {WHITE: 0, BLACK: 0, "draws": 0}

if "game_state" not in st.session_state:
    reset_game()
    st.session_state.scores = {WHITE: 0, BLACK: 0, "draws": 0}

# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.markdown("### ⚛ Settings")
    mode = st.radio("Mode", ["Player vs AI", "Player vs Player", "AI vs AI"], index=0)
    difficulty = st.select_slider("AI Difficulty", ["Easy", "Medium", "Hard"], value="Medium")

    if mode == "Player vs AI":
        human_side = st.radio("You play as", ["White (W)", "Black (B)"])
        human_player = WHITE if "White" in human_side else BLACK
        ai_player = 3 - human_player
    else:
        human_player = None
        ai_player = None

    st.markdown("---")
    st.markdown("### 📖 Rules")
    st.markdown("""
<div class="rule-card">
<b>Grid:</b> 2×2 = 4 squares. Pool starts with 4 double-sided tokens.
</div>
<div class="rule-card">
<b>Place:</b> Take from pool → place on empty square, your color up.
</div>
<div class="rule-card">
<b>Flip:</b> Turn any opponent's token to show your color.
</div>
<div class="rule-card">
<b>Win:</b> All 4 squares show your color at end of your turn.
</div>
<div class="rule-card">
<b>Trap:</b> If you have no valid moves, you lose.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 New Game", use_container_width=True, type="primary"):
        sc = st.session_state.scores
        reset_game()
        st.session_state.scores = sc
        st.rerun()
    if st.button("🗑 Reset Scores", use_container_width=True):
        st.session_state.scores = {WHITE: 0, BLACK: 0, "draws": 0}
        st.rerun()

# ============================================================================
# Main
# ============================================================================
gs = st.session_state.game_state
scores = st.session_state.scores

# Score Row
sc1, sc2, sc3 = st.columns(3)
with sc1:
    st.markdown(f"""<div class="stat-card">
        <div class="val" style="color:#caf0f8">{scores[WHITE]}</div>
        <div class="lbl">⬜ White Wins</div></div>""", unsafe_allow_html=True)
with sc2:
    st.markdown(f"""<div class="stat-card">
        <div class="val" style="color:#48cae4">{scores.get('draws',0)}</div>
        <div class="lbl">⚛ Draws</div></div>""", unsafe_allow_html=True)
with sc3:
    st.markdown(f"""<div class="stat-card">
        <div class="val" style="color:#023e8a">{scores[BLACK]}</div>
        <div class="lbl">⬛ Black Wins</div></div>""", unsafe_allow_html=True)

st.markdown("")

left_col, right_col = st.columns([3, 2])

with left_col:
    if gs["game_over"]:
        if gs["winner"] == WHITE:
            st.markdown('<div class="win-banner">⬜ WHITE WINS — all 4 squares claimed!</div>', unsafe_allow_html=True)
        elif gs["winner"] == BLACK:
            st.markdown('<div class="win-banner" style="background:linear-gradient(90deg,#03045e,#023e8a)">⬛ BLACK WINS — all 4 squares claimed!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="win-banner" style="background:linear-gradient(90deg,#1b4332,#2d6a4f)">⚛ DRAW — no moves remain</div>', unsafe_allow_html=True)
    else:
        turn_label = "White (W)" if gs["turn"] == WHITE else "Black (B)"
        symbol = "⬜" if gs["turn"] == WHITE else "⬛"
        st.markdown(f'<div class="turn-indicator">{symbol} {turn_label}\'s Turn · Turn {gs["turn_count"] + 1} · Pool: {gs["pool"]}/4 remaining</div>', unsafe_allow_html=True)

    fig = draw_board(gs)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Last action display
    if gs["last_action"]:
        st.caption(f"Last: {gs['last_action']}")

    # Human action panel
    if not gs["game_over"]:
        is_human_turn = (
            mode == "Player vs Player" or
            (mode == "Player vs AI" and gs["turn"] == human_player)
        )

        if is_human_turn:
            actions = get_valid_actions(gs)
            place_acts = [(t, i) for t, i in actions if t == "place"]
            flip_acts  = [(t, i) for t, i in actions if t == "flip"]

            st.markdown('<div class="action-section">', unsafe_allow_html=True)
            st.markdown("**🎯 Place from pool** (token goes on empty square)")
            if place_acts:
                pcols = st.columns(len(place_acts))
                for ci, (t, i) in enumerate(place_acts):
                    with pcols[ci]:
                        if st.button(f"Place → {_pos_name(i)}", key=f"place_{i}"):
                            new_state = apply_action(gs, (t, i))
                            if new_state["game_over"] and new_state["winner"]:
                                scores[new_state["winner"]] = scores.get(new_state["winner"], 0) + 1
                            elif new_state["game_over"] and not new_state["winner"]:
                                scores["draws"] = scores.get("draws", 0) + 1
                            st.session_state.game_state = new_state
                            st.rerun()
            else:
                st.caption("No empty squares to place on.")

            st.markdown("**🔄 Flip opponent's token**")
            if flip_acts:
                fcols = st.columns(len(flip_acts))
                for ci, (t, i) in enumerate(flip_acts):
                    with fcols[ci]:
                        if st.button(f"Flip {_pos_name(i)}", key=f"flip_{i}"):
                            new_state = apply_action(gs, (t, i))
                            if new_state["game_over"] and new_state["winner"]:
                                scores[new_state["winner"]] = scores.get(new_state["winner"], 0) + 1
                            elif new_state["game_over"] and not new_state["winner"]:
                                scores["draws"] = scores.get("draws", 0) + 1
                            st.session_state.game_state = new_state
                            st.rerun()
            else:
                st.caption("No opponent tokens to flip.")
            st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown("### 🔬 Analysis")

    # Grid ownership breakdown
    grid = gs["grid"]
    w_count = grid.count(WHITE)
    b_count = grid.count(BLACK)
    e_count = grid.count(EMPTY)

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("⬜ White", f"{w_count}/4", delta=f"Need {max(0,4-w_count)} more")
        st.metric("Pool Left", gs["pool"])
    with col_b:
        st.metric("⬛ Black", f"{b_count}/4", delta=f"Need {max(0,4-b_count)} more")
        st.metric("Empty Sq", e_count)

    # Danger meter
    st.markdown("**Tension Meter**")
    max_ctrl = max(w_count, b_count)
    if max_ctrl >= 3:
        leader = "White" if w_count >= b_count else "Black"
        st.warning(f"⚠️ {leader} needs just {4 - max_ctrl} more square(s) to win!")
    elif max_ctrl == 2:
        st.info("Both players have 2 squares — critical phase.")
    else:
        st.success("Early game — board still open.")

    # AI heat map
    st.markdown("**AI Value Map** (per square)")
    ai_viz_player = ai_player if ai_player else WHITE
    fig2 = draw_probability_map(gs, ai_viz_player)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
    st.caption("Heat = how much AI values owning that square")

    # Move log
    st.markdown("**Move Log**")
    if gs["history"]:
        log = []
        for i, (atype, idx, player) in enumerate(gs["history"][-8:]):
            sym = "⬜" if player == WHITE else "⬛"
            verb = "placed" if atype == "place" else "flipped"
            log.append(f"`{i+1}.` {sym} {verb} **{_pos_name(idx)}**")
        st.markdown("\n".join(log))
    else:
        st.caption("No moves yet.")

    st.markdown("---")
    st.markdown("**State Complexity**")
    # 3^4 = 81 states
    occupied = sum(1 for v in grid if v != EMPTY)
    import math as _math
    states_visited = min(81, 3 ** occupied)
    st.caption(f"States reachable so far: ~{states_visited} / 81")
    st.progress(states_visited / 81)

# ============================================================================
# AI turn trigger
# ============================================================================
if not gs["game_over"]:
    if mode == "Player vs AI" and gs["turn"] == ai_player:
        with st.spinner("⚛ Quantum AI calculating..."):
            import time; time.sleep(0.4)
            action = ai_choose(gs, ai_player, difficulty)
            if action:
                new_state = apply_action(gs, action)
                if new_state["game_over"] and new_state["winner"]:
                    scores[new_state["winner"]] = scores.get(new_state["winner"], 0) + 1
                elif new_state["game_over"] and not new_state["winner"]:
                    scores["draws"] = scores.get("draws", 0) + 1
                st.session_state.game_state = new_state
                st.rerun()

    elif mode == "AI vs AI":
        import time; time.sleep(0.6)
        current = gs["turn"]
        action = ai_choose(gs, current, difficulty)
        if action:
            new_state = apply_action(gs, action)
            if new_state["game_over"] and new_state["winner"]:
                scores[new_state["winner"]] = scores.get(new_state["winner"], 0) + 1
            elif new_state["game_over"] and not new_state["winner"]:
                scores["draws"] = scores.get("draws", 0) + 1
            st.session_state.game_state = new_state
            st.rerun()

# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.caption("Quantum Match · 2×2 grid · Flip or place mechanics · State space: 81 configurations · Minimax AI")
