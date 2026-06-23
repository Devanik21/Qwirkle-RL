import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon
import math
from copy import deepcopy
import random

# ============================================================================
# Page Config
# ============================================================================
st.set_page_config(
    page_title="Hex-Line",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⬡"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.title-block {
    background: linear-gradient(135deg, #0d0d0d 0%, #1a0a2e 50%, #0d0d0d 100%);
    border: 1px solid #7c3aed;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    box-shadow: 0 0 40px rgba(124,58,237,0.18);
}
.title-block h1 { color: #a78bfa; font-size: 2.4rem; margin: 0; letter-spacing: -1px; font-weight: 700; }
.title-block p  { color: #c4b5fd; font-size: 0.97rem; margin: 6px 0 0; opacity: 0.85; }

.rule-card {
    background: #0f0f1a;
    border: 1px solid #312e5a;
    border-left: 3px solid #7c3aed;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    color: #d4d0f0;
    font-size: 0.88rem;
    line-height: 1.6;
}
.stat-card {
    background: #0f0f1a;
    border: 1px solid #312e5a;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    color: #a78bfa;
}
.stat-card .val { font-size: 2rem; font-weight: 700; color: #c4b5fd; }
.stat-card .lbl { font-size: 0.78rem; color: #6d6a96; margin-top: 2px; }

.win-banner {
    background: linear-gradient(90deg, #4c1d95, #7c3aed);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
    color: #fff;
    margin: 16px 0;
    box-shadow: 0 0 30px rgba(124,58,237,0.4);
}
.turn-indicator {
    background: #0f0f1a;
    border: 1px solid #312e5a;
    border-radius: 10px;
    padding: 12px 18px;
    margin: 10px 0;
    font-size: 1rem;
    color: #c4b5fd;
    text-align: center;
}
stButton > button {
    background: #1a103a !important;
    border: 1px solid #7c3aed !important;
    color: #c4b5fd !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-block">
  <h1>⬡ Hex-Line</h1>
  <p>Minimalist connection strategy · 7 hexagons · Zero ties possible · Full Minimax AI</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Game Logic
# ============================================================================

# Board: 7 positions — index 0 = center, 1–6 = outer ring (clockwise from top)
# Lines through center: (1,0,4), (2,0,5), (3,0,6)
# Outer ring wrap: positions 1–6

CENTER = 0
OUTER = list(range(1, 7))
LINES_THROUGH_CENTER = [(1, 0, 4), (2, 0, 5), (3, 0, 6)]

EMPTY = 0
P1 = 1  # Purple
P2 = 2  # Orange

def initial_state():
    return [EMPTY] * 7

def get_valid_moves(state):
    return [i for i, v in enumerate(state) if v == EMPTY]

def check_winner(state):
    # Win condition 1: straight line of 3 through center
    for a, b, c in LINES_THROUGH_CENTER:
        if state[a] != EMPTY and state[a] == state[b] == state[c]:
            return state[a], (a, b, c)
    # Win condition 2: occupy 4+ of the 6 outer hexes
    for player in [P1, P2]:
        outer_count = sum(1 for i in OUTER if state[i] == player)
        if outer_count >= 4:
            return player, None
    return None, None

def is_terminal(state):
    winner, _ = check_winner(state)
    if winner:
        return True
    return len(get_valid_moves(state)) == 0

def score_state(state, player):
    winner, _ = check_winner(state)
    if winner == player:
        return 1000
    if winner == (3 - player):
        return -1000
    return 0

def evaluate_heuristic(state, player):
    """Heuristic: center control, line threats, outer dominance"""
    opp = 3 - player
    score = 0

    # Center bonus
    if state[CENTER] == player:
        score += 30
    elif state[CENTER] == opp:
        score -= 30

    # Line threat scoring
    for a, b, c in LINES_THROUGH_CENTER:
        line = [state[a], state[b], state[c]]
        p_count = line.count(player)
        o_count = line.count(opp)
        if o_count == 0:
            score += p_count * 15
        if p_count == 0:
            score -= o_count * 15
        # Two in a line threat
        if p_count == 2 and o_count == 0:
            score += 40
        if o_count == 2 and p_count == 0:
            score -= 40

    # Outer dominance
    p_outer = sum(1 for i in OUTER if state[i] == player)
    o_outer = sum(1 for i in OUTER if state[i] == opp)
    score += p_outer * 10
    score -= o_outer * 10

    # Near-win outer bonus
    if p_outer >= 3:
        score += (p_outer - 2) * 25
    if o_outer >= 3:
        score -= (o_outer - 2) * 25

    return score

def minimax(state, depth, alpha, beta, maximizing, ai_player):
    winner, _ = check_winner(state)
    if winner == ai_player:
        return 1000 + depth, None
    if winner == (3 - ai_player):
        return -1000 - depth, None
    moves = get_valid_moves(state)
    if not moves or depth == 0:
        return evaluate_heuristic(state, ai_player), None

    best_move = None
    if maximizing:
        best_val = -float('inf')
        for m in moves:
            ns = state[:]
            ns[m] = ai_player
            val, _ = minimax(ns, depth - 1, alpha, beta, False, ai_player)
            if val > best_val:
                best_val, best_move = val, m
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return best_val, best_move
    else:
        opp = 3 - ai_player
        best_val = float('inf')
        for m in moves:
            ns = state[:]
            ns[m] = opp
            val, _ = minimax(ns, depth - 1, alpha, beta, True, ai_player)
            if val < best_val:
                best_val, best_move = val, m
            beta = min(beta, val)
            if beta <= alpha:
                break
        return best_val, best_move

def ai_move(state, ai_player, difficulty):
    depths = {"Easy": 2, "Medium": 4, "Hard": 7}
    d = depths[difficulty]
    if difficulty == "Easy" and random.random() < 0.35:
        moves = get_valid_moves(state)
        return random.choice(moves) if moves else None
    _, move = minimax(state, d, -float('inf'), float('inf'), True, ai_player)
    return move

# ============================================================================
# Visualization
# ============================================================================

# Hex centers in pixel space: center + 6 outer
# Using flat-top hexagons. Outer ring radius = 1.8 units.
HEX_POSITIONS = {
    0: (0.0, 0.0),           # center
    1: (0.0, 1.8),           # top
    2: (1.558, 0.9),         # top-right
    3: (1.558, -0.9),        # bottom-right
    4: (0.0, -1.8),          # bottom
    5: (-1.558, -0.9),       # bottom-left
    6: (-1.558, 0.9),        # top-left
}
HEX_LABELS = {0: "C", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6"}

P1_COLOR   = "#7c3aed"  # purple
P2_COLOR   = "#f97316"  # orange
EMPTY_COLOR = "#1e1b3a"
BORDER_COLOR = "#4c1d95"
HIGHLIGHT_COLOR = "#a78bfa"
WIN_LINE_COLOR = "#fbbf24"

def draw_hex(ax, cx, cy, size, facecolor, edgecolor, alpha=1.0, lw=2.5):
    hex_patch = RegularPolygon(
        (cx, cy), numVertices=6, radius=size,
        orientation=math.pi / 6,
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, alpha=alpha, zorder=2
    )
    ax.add_patch(hex_patch)

def draw_board(state, win_line=None, hover=None, mode="pvp"):
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    HEX_SIZE = 0.78
    label_colors = {EMPTY: "#4a4570", P1: "#c4b5fd", P2: "#fdba74"}

    # Draw lines first (the 3 axes through center)
    for a, b, c in LINES_THROUGH_CENTER:
        x0, y0 = HEX_POSITIONS[a]
        x1, y1 = HEX_POSITIONS[c]
        ax.plot([x0, x1], [y0, y1], color="#2d2a52", lw=1.2, zorder=1, alpha=0.6)

    if win_line:
        x0, y0 = HEX_POSITIONS[win_line[0]]
        x1, y1 = HEX_POSITIONS[win_line[2]]
        ax.plot([x0, x1], [y0, y1], color=WIN_LINE_COLOR, lw=4, zorder=3, alpha=0.85)

    for idx in range(7):
        cx, cy = HEX_POSITIONS[idx]
        piece = state[idx]

        if piece == P1:
            face = P1_COLOR
            ec   = "#a78bfa"
        elif piece == P2:
            face = P2_COLOR
            ec   = "#fb923c"
        else:
            face = EMPTY_COLOR
            ec   = BORDER_COLOR

        lw = 3.5 if (win_line and idx in win_line) else 2.2
        draw_hex(ax, cx, cy, HEX_SIZE, face, ec, lw=lw)

        # Position label
        lc = label_colors[piece]
        sym = "●" if piece == P1 else ("●" if piece == P2 else str(idx))
        fc = "#c4b5fd" if piece == P1 else ("#fdba74" if piece == P2 else "#4a4570")
        ax.text(cx, cy, sym, ha="center", va="center",
                fontsize=22 if piece != EMPTY else 14,
                color=fc, fontweight="bold", zorder=4)

        if piece == EMPTY:
            ax.text(cx, cy - 0.28, HEX_LABELS[idx], ha="center", va="center",
                    fontsize=9, color="#3d3a60", zorder=4)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.axis("off")

    # Legend
    p1p = mpatches.Patch(color=P1_COLOR, label="Player 1 (Purple)")
    p2p = mpatches.Patch(color=P2_COLOR, label="Player 2 (Orange)")
    ax.legend(handles=[p1p, p2p], loc="lower center",
              facecolor="#0d0d0d", edgecolor="#312e5a",
              labelcolor="white", fontsize=9, ncol=2,
              bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    return fig

# ============================================================================
# Session State Init
# ============================================================================

def reset_game():
    st.session_state.board = initial_state()
    st.session_state.turn = P1
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.win_line = None
    st.session_state.move_count = 0
    st.session_state.history = []
    st.session_state.scores = st.session_state.get("scores", {P1: 0, P2: 0, "draws": 0})

if "board" not in st.session_state:
    reset_game()
    st.session_state.scores = {P1: 0, P2: 0, "draws": 0}

# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.markdown("### ⬡ Game Settings")
    mode = st.radio("Mode", ["Player vs AI", "Player vs Player", "AI vs AI"], index=0)
    difficulty = st.select_slider("AI Difficulty", ["Easy", "Medium", "Hard"], value="Medium")
    if mode == "Player vs AI":
        human_player = st.radio("You play as", ["Player 1 (Purple)", "Player 2 (Orange)"])
        human_id = P1 if "1" in human_player else P2
        ai_id = 3 - human_id
    else:
        human_id = None
        ai_id = None

    st.markdown("---")
    st.markdown("### 📖 Rules")
    st.markdown("""
<div class="rule-card">
<b>Board:</b> 7 hexagons — 1 center + 6 outer ring.
</div>
<div class="rule-card">
<b>Win A:</b> Straight line of 3 through the center hex.
</div>
<div class="rule-card">
<b>Win B:</b> Occupy 4 of the 6 outer hexagons.
</div>
<div class="rule-card">
<b>No ties possible</b> — the geometry guarantees it.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 New Game", use_container_width=True, type="primary"):
        board_backup = st.session_state.scores
        reset_game()
        st.session_state.scores = board_backup
        st.rerun()
    if st.button("🗑 Reset Scores", use_container_width=True):
        st.session_state.scores = {P1: 0, P2: 0, "draws": 0}
        st.rerun()

# ============================================================================
# Main Layout
# ============================================================================
board = st.session_state.board
scores = st.session_state.scores

# Score row
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""<div class="stat-card">
        <div class="val" style="color:#c4b5fd">{scores[P1]}</div>
        <div class="lbl">🟣 Player 1 Wins</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="stat-card">
        <div class="val" style="color:#94a3b8">{scores.get('draws',0)}</div>
        <div class="lbl">⬡ Draws</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="stat-card">
        <div class="val" style="color:#fdba74">{scores[P2]}</div>
        <div class="lbl">🟠 Player 2 Wins</div></div>""", unsafe_allow_html=True)

st.markdown("")

left_col, right_col = st.columns([3, 2])

with left_col:
    # Status
    if st.session_state.game_over:
        if st.session_state.winner:
            w = st.session_state.winner
            color = "#c4b5fd" if w == P1 else "#fdba74"
            label = "Player 1 (Purple)" if w == P1 else "Player 2 (Orange)"
            st.markdown(f'<div class="win-banner">🏆 {label} Wins!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="win-banner" style="background:linear-gradient(90deg,#1e3a5f,#2563eb)">🤝 Draw!</div>', unsafe_allow_html=True)
    else:
        turn_label = "Player 1 (Purple) 🟣" if st.session_state.turn == P1 else "Player 2 (Orange) 🟠"
        st.markdown(f'<div class="turn-indicator">Turn → <b>{turn_label}</b> · Move #{st.session_state.move_count + 1}</div>', unsafe_allow_html=True)

    # Board render
    fig = draw_board(board, win_line=st.session_state.win_line)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Move buttons (human input)
    if not st.session_state.game_over:
        show_buttons = False
        if mode == "Player vs Player":
            show_buttons = True
        elif mode == "Player vs AI" and st.session_state.turn == human_id:
            show_buttons = True

        if show_buttons:
            valid = get_valid_moves(board)
            st.markdown("**Pick a hex to place your token:**")
            btn_cols = st.columns(7)
            pos_labels = {0: "C·0", 1: "T·1", 2: "TR·2", 3: "BR·3", 4: "B·4", 5: "BL·5", 6: "TL·6"}
            for pos in range(7):
                with btn_cols[pos]:
                    if pos in valid:
                        if st.button(pos_labels[pos], key=f"move_{pos}"):
                            board[pos] = st.session_state.turn
                            st.session_state.move_count += 1
                            st.session_state.history.append(pos)
                            winner, win_line = check_winner(board)
                            if winner:
                                st.session_state.game_over = True
                                st.session_state.winner = winner
                                st.session_state.win_line = win_line
                                scores[winner] += 1
                            elif not get_valid_moves(board):
                                st.session_state.game_over = True
                                st.session_state.winner = None
                                scores["draws"] = scores.get("draws", 0) + 1
                            else:
                                st.session_state.turn = 3 - st.session_state.turn
                            st.rerun()
                    else:
                        st.button(pos_labels[pos], key=f"move_{pos}_off", disabled=True)

with right_col:
    st.markdown("### 🧠 Game Intel")

    # Move history
    if st.session_state.history:
        st.markdown("**Move Log**")
        log_lines = []
        for i, pos in enumerate(st.session_state.history):
            player = P1 if i % 2 == 0 else P2
            sym = "🟣" if player == P1 else "🟠"
            pos_labels2 = {0: "Center", 1: "Top", 2: "Top-R", 3: "Bot-R", 4: "Bottom", 5: "Bot-L", 6: "Top-L"}
            log_lines.append(f"`{i+1}.` {sym} → **{pos_labels2[pos]}** (hex {pos})")
        st.markdown("\n".join(log_lines[-10:]))

    st.markdown("---")

    # Live analysis
    st.markdown("**Position Analysis**")
    p1_outer = sum(1 for i in OUTER if board[i] == P1)
    p2_outer = sum(1 for i in OUTER if board[i] == P2)
    center_owner = board[CENTER]

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("🟣 P1 Outer", f"{p1_outer}/6", delta=f"Need {max(0,4-p1_outer)} more")
        st.metric("Center", "🟣" if center_owner==P1 else ("🟠" if center_owner==P2 else "—"))
    with col_b:
        st.metric("🟠 P2 Outer", f"{p2_outer}/6", delta=f"Need {max(0,4-p2_outer)} more")
        st.metric("Moves Left", len(get_valid_moves(board)))

    # Threat detector
    st.markdown("**Line Threats**")
    threat_info = []
    line_names = ["Top↔Bottom", "TRight↔BLeft", "BRight↔TLeft"]
    for i, (a, b, c) in enumerate(LINES_THROUGH_CENTER):
        line = [board[a], board[b], board[c]]
        p1c, p2c = line.count(P1), line.count(P2)
        if p1c == 2 and p2c == 0:
            threat_info.append(f"🟣 **P1 threatening** {line_names[i]}")
        elif p2c == 2 and p1c == 0:
            threat_info.append(f"🟠 **P2 threatening** {line_names[i]}")
        elif p1c > 0 and p2c > 0:
            threat_info.append(f"⬡ {line_names[i]}: blocked")
        else:
            threat_info.append(f"○ {line_names[i]}: open")
    for t in threat_info:
        st.markdown(t)

    st.markdown("---")
    st.markdown("**State Space**")
    st.caption(f"Positions occupied: {sum(1 for x in board if x != EMPTY)}/7")
    filled = sum(1 for x in board if x != EMPTY)
    pct = int((filled / 7) * 100)
    st.progress(pct / 100, text=f"{pct}% filled")

# ============================================================================
# AI Turn Logic
# ============================================================================
if not st.session_state.game_over:
    if mode == "Player vs AI" and st.session_state.turn == ai_id:
        with st.spinner("🤖 AI thinking..."):
            import time
            time.sleep(0.4)
            move = ai_move(board, ai_id, difficulty)
            if move is not None:
                board[move] = ai_id
                st.session_state.move_count += 1
                st.session_state.history.append(move)
                winner, win_line = check_winner(board)
                if winner:
                    st.session_state.game_over = True
                    st.session_state.winner = winner
                    st.session_state.win_line = win_line
                    scores[winner] += 1
                elif not get_valid_moves(board):
                    st.session_state.game_over = True
                    scores["draws"] = scores.get("draws", 0) + 1
                else:
                    st.session_state.turn = 3 - st.session_state.turn
                st.rerun()

    elif mode == "AI vs AI" and not st.session_state.game_over:
        import time
        time.sleep(0.5)
        current = st.session_state.turn
        move = ai_move(board, current, difficulty)
        if move is not None:
            board[move] = current
            st.session_state.move_count += 1
            st.session_state.history.append(move)
            winner, win_line = check_winner(board)
            if winner:
                st.session_state.game_over = True
                st.session_state.winner = winner
                st.session_state.win_line = win_line
                scores[winner] += 1
            elif not get_valid_moves(board):
                st.session_state.game_over = True
                scores["draws"] = scores.get("draws", 0) + 1
            else:
                st.session_state.turn = 3 - current
            st.rerun()

# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.caption("Hex-Line · 7-hex geometry · Minimax AI with alpha-beta pruning · State space: ~700 legal positions")
