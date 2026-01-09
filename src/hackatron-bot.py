import sys, random, time
from collections import deque
from src.types import GameState, LEFT, UP, RIGHT, DOWN

# Pre-compute movement deltas: (direction_id, row_delta, col_delta)
MOVES_FLAT = [
    (LEFT, 0, -1),      # Move left (column - 1)
    (UP, -1, 0),        # Move up (row - 1)
    (RIGHT, 0, 1),      # Move right (column + 1)
    (DOWN, 1, 0)        # Move down (row + 1)
]

# === SEARCH CONFIGURATION ===
BASE_DEPTH = 4              # Starting search depth for minimax
MAX_DEPTH = 50              # Maximum depth to search (with Late Move Reduction)
TIME_LIMIT = 0.050          # Total time available per move (50ms)
TIME_BUFFER = 0.005         # Safety buffer to avoid timeout (5ms)
RANDOM_TIE = 0.005          # Random noise for tie-breaking moves

# === TRANSPOSITION TABLE & MOVE ORDERING ===
TRANSPOSITION_TABLE = {}    # Store previously evaluated positions: hash -> (value, depth, flag, best_move)
MOVE_HISTORY = {LEFT: 0, UP: 0, RIGHT: 0, DOWN: 0}  # Track move success frequency
KILLER_MOVES = [[None, None] for _ in range(MAX_DEPTH + 20)]  # Best non-capture moves per depth

# === ZOBRIST HASHING ===
# Zobrist hashing creates unique 64-bit hashes for board positions
ZOBRIST_BOARD = [[random.getrandbits(64) for _ in range(30)] for _ in range(30)]  # Hash for board cells
ZOBRIST_PLAYER1_POS = [[random.getrandbits(64) for _ in range(30)] for _ in range(30)]  # Hash for my position
ZOBRIST_PLAYER2_POS = [[random.getrandbits(64) for _ in range(30)] for _ in range(30)]  # Hash for opponent position

# === CONSTANTS ===
INFINITY = 1e12             # Large value representing infinite score
node_count = 0              # Counter for nodes evaluated (for time management)

def voronoi_territory(board, my_row, my_col, opp_row, opp_col):
    """
    Use BFS (Flood Fill) to calculate territory control between two players.
    
    Returns:
        tuple: (my_territory_count, opp_territory_count, are_territories_connected)
    """
    board_size = len(board)
    
    # Track which cells each player has visited
    my_visited = {(my_row, my_col)}
    opp_visited = {(opp_row, opp_col)}
    
    # Queue for BFS expansion from each player's position
    my_queue = deque([(my_row, my_col)])
    opp_queue = deque([(opp_row, opp_col)])
    
    # Count of empty cells claimed by each player
    my_territory_count = 0
    opp_territory_count = 0
    
    # Flag: whether the territories touch (game still competitive)
    territories_connected = False
    
    # Expand territories until queues are empty
    while my_queue or opp_queue:
        # === EXPAND MY TERRITORY ===
        if my_queue:
            # Process all cells at current layer
            for _ in range(len(my_queue)):
                current_row, current_col = my_queue.popleft()
                
                # Check all 4 adjacent cells
                for _, delta_row, delta_col in MOVES_FLAT:
                    neighbor_row = current_row + delta_row
                    neighbor_col = current_col + delta_col
                    
                    # Verify neighbor is within bounds and is empty
                    if 0 <= neighbor_row < board_size and 0 <= neighbor_col < board_size and board[neighbor_row][neighbor_col] == 0:
                        # If opponent already claimed it: territories are connected
                        if (neighbor_row, neighbor_col) in opp_visited:
                            territories_connected = True
                        # If unclaimed, add to my territory
                        elif (neighbor_row, neighbor_col) not in my_visited:
                            my_visited.add((neighbor_row, neighbor_col))
                            my_territory_count += 1
                            my_queue.append((neighbor_row, neighbor_col))

        # === EXPAND OPPONENT TERRITORY ===
        if opp_queue:
            # Process all cells at current layer
            for _ in range(len(opp_queue)):
                current_row, current_col = opp_queue.popleft()
                
                # Check all 4 adjacent cells
                for _, delta_row, delta_col in MOVES_FLAT:
                    neighbor_row = current_row + delta_row
                    neighbor_col = current_col + delta_col
                    
                    # Verify neighbor is within bounds and is empty
                    if 0 <= neighbor_row < board_size and 0 <= neighbor_col < board_size and board[neighbor_row][neighbor_col] == 0:
                        # If I already claimed it: territories are connected
                        if (neighbor_row, neighbor_col) in my_visited:
                            territories_connected = True
                        # If unclaimed, add to opponent territory
                        elif (neighbor_row, neighbor_col) not in opp_visited:
                            opp_visited.add((neighbor_row, neighbor_col))
                            opp_territory_count += 1
                            opp_queue.append((neighbor_row, neighbor_col))

    return my_territory_count, opp_territory_count, territories_connected

def evaluate(my_row, my_col, opp_row, opp_col, board):
    """
    Evaluate the current board position from my perspective.
    Higher score = better for me, lower score = better for opponent.
    """
    board_size = len(board)
    
    # === CALCULATE TERRITORY CONTROL ===
    my_territory, opp_territory, are_connected = voronoi_territory(board, my_row, my_col, opp_row, opp_col)
    
    # === CALCULATE IMMEDIATE MOBILITY (move options) ===
    my_mobility = 0
    for _, delta_row, delta_col in MOVES_FLAT:
        neighbor_row = my_row + delta_row
        neighbor_col = my_col + delta_col
        # Count valid empty neighbors
        if 0 <= neighbor_row < board_size and 0 <= neighbor_col < board_size and board[neighbor_row][neighbor_col] == 0:
            my_mobility += 1

    opp_mobility = 0
    for _, delta_row, delta_col in MOVES_FLAT:
        neighbor_row = opp_row + delta_row
        neighbor_col = opp_col + delta_col
        # Count valid empty neighbors
        if 0 <= neighbor_row < board_size and 0 <= neighbor_col < board_size and board[neighbor_row][neighbor_col] == 0:
            opp_mobility += 1
    
    # === ENDGAME: SEPARATED TERRITORIES ===
    # If territories don't touch, winner is determined by who has more space
    if not are_connected:
        if my_territory > opp_territory:
            # I win by a landslide - add huge bonus
            return 1000000 + (my_territory * 100)
        elif my_territory < opp_territory:
            # I lose by a landslide - add huge penalty
            return -1000000 - (opp_territory * 100)
        else:
            # Tie in space, use mobility as tiebreaker
            return (my_mobility - opp_mobility) * 50

    # === CONNECTED STATE: TERRITORIES STILL COMPETE ===
    score = 0
    
    # Territory difference is the strongest evaluation factor
    score = (my_territory - opp_territory) * 10.0
    
    # Bonus for having more move options
    score += (my_mobility - opp_mobility) * 20.0
    
    # Aggression bonus: being closer to opponent cuts off their space
    # But only if I have the territory advantage (don't be reckless)
    manhattan_distance = abs(my_row - opp_row) + abs(my_col - opp_col)
    if my_territory >= opp_territory:
        score += (20.0 / (manhattan_distance + 1))
    
    # Minor penalty for being far from board center (centralization helps control)
    board_center = board_size / 2.0
    distance_to_center = abs(my_row - board_center) + abs(my_col - board_center)
    score -= distance_to_center * 0.5
    
    # PANIC if space is critically low (avoid getting trapped)
    if my_territory < 15:
        score -= 5000
    
    return score

def order_moves(my_row, my_col, board, current_depth, principal_variation_move=None):
    """
    Order moves by heuristic quality for alpha-beta pruning efficiency.
    Better moves are evaluated first, allowing more pruning.
    """
    board_size = len(board)
    scored_moves = []
    
    # === SCORE EACH LEGAL MOVE ===
    for move_direction, delta_row, delta_col in MOVES_FLAT:
        new_row = my_row + delta_row
        new_col = my_col + delta_col
        
        # Skip illegal moves (out of bounds or blocked)
        if not (0 <= new_row < board_size and 0 <= new_col < board_size and board[new_row][new_col] == 0):
            continue
        
        # === LOOKAHEAD: COUNT FUTURE MOVES FROM THIS POSITION ===
        # Prefer moves that leave many options
        available_neighbors = 0
        for _, next_delta_row, next_delta_col in MOVES_FLAT:
            next_row = new_row + next_delta_row
            next_col = new_col + next_delta_col
            if 0 <= next_row < board_size and 0 <= next_col < board_size and board[next_row][next_col] == 0:
                available_neighbors += 1
        
        # === APPLY MOVE ORDERING HEURISTICS ===
        # Principal Variation (PV) moves are most likely to be best
        pv_bonus = 20000 if principal_variation_move == move_direction else 0
        
        # Killer moves worked well at this depth in other branches
        killer_bonus = 10000 if move_direction in KILLER_MOVES[current_depth] else 0
        
        # History heuristic: moves that caused cutoffs before are likely good
        history_score = MOVE_HISTORY.get(move_direction, 0)
        
        # Dead ends (0 available moves) are terrible - avoid them
        dead_end_penalty = -50000 if available_neighbors == 0 else 0
        
        # Combine heuristics into total sort score
        total_score = (available_neighbors * 50) + history_score + killer_bonus + pv_bonus + dead_end_penalty
        scored_moves.append((total_score, move_direction, new_row, new_col))
    
    # Sort by score (highest first = best moves first)
    scored_moves.sort(reverse=True, key=lambda x: x[0])
    return scored_moves

def minimax(my_row, my_col, opp_row, opp_col, board, current_depth, alpha_bound, beta_bound, is_my_turn, current_ply, deadline, zobrist_hash):
    """
    Minimax with Alpha-Beta pruning, Transposition Tables, and Move Ordering.
    
    - is_my_turn: True = maximize, False = minimize
    - current_ply: depth from root (for killer moves)
    - zobrist_hash: unique board state hash
    """
    global node_count
    node_count += 1
    
    # === TIME MANAGEMENT ===
    # Check timeout frequently (every 128 nodes = every 2^7 nodes)
    if (node_count & 127) == 0:
        if time.time() >= deadline:
            return evaluate(my_row, my_col, opp_row, opp_col, board)

    # === BASE CASE: SEARCH DEPTH LIMIT ===
    if current_depth <= 0:
        return evaluate(my_row, my_col, opp_row, opp_col, board)

    # === TRANSPOSITION TABLE LOOKUP ===
    # Check if this position was already evaluated
    if zobrist_hash in TRANSPOSITION_TABLE:
        stored_value, stored_depth, stored_flag, best_previous_move = TRANSPOSITION_TABLE[zobrist_hash]
        # Only use TT entry if it was evaluated to sufficient depth
        if stored_depth >= current_depth:
            if stored_flag == 0:
                return stored_value  # Exact value
            if stored_flag == -1 and stored_value <= alpha_bound:
                return stored_value  # Upper bound (fail-low)
            if stored_flag == 1 and stored_value >= beta_bound:
                return stored_value  # Lower bound (fail-high)
    else:
        best_previous_move = None

    board_size = len(board)

    # === MAXIMIZING PLAYER (MY TURN) ===
    if is_my_turn:
        # Get all legal moves, ordered by heuristic quality
        legal_moves = order_moves(my_row, my_col, board, current_ply, best_previous_move)
        
        # No legal moves = I'm trapped = loss
        if not legal_moves:
            return -200000.0 + current_ply  # Negative because I lose

        best_value = -INFINITY
        best_move_found = None
        
        # === SEARCH EACH MOVE ===
        for move_index, (_, move_direction, new_row, new_col) in enumerate(legal_moves):
            # === MAKE MOVE ===
            board[new_row][new_col] = 1
            new_zobrist_hash = zobrist_hash ^ ZOBRIST_PLAYER1_POS[my_row][my_col] ^ ZOBRIST_PLAYER1_POS[new_row][new_col] ^ ZOBRIST_BOARD[new_row][new_col]
            
            # === LATE MOVE REDUCTION (LMR) ===
            # If searching deep and this isn't a promising move, search shallower first
            depth_reduction = 0
            if current_depth >= 3 and move_index >= 3 and move_direction not in KILLER_MOVES[current_ply]:
                depth_reduction = 1
            
            # Recursively search opponent's response
            value = minimax(new_row, new_col, opp_row, opp_col, board, current_depth - 1 - depth_reduction, 
                           alpha_bound, beta_bound, False, current_ply + 1, deadline, new_zobrist_hash)
            
            # === LATE MOVE REDUCTION RE-SEARCH ===
            # If reduction was used and move is better than expected, search full depth
            if depth_reduction > 0 and value > alpha_bound:
                value = minimax(new_row, new_col, opp_row, opp_col, board, current_depth - 1,
                               alpha_bound, beta_bound, False, current_ply + 1, deadline, new_zobrist_hash)

            # === UNDO MOVE ===
            board[new_row][new_col] = 0
            
            # === UPDATE BEST VALUE ===
            if value > best_value:
                best_value = value
                best_move_found = move_direction
            
            # === ALPHA-BETA PRUNING ===
            alpha_bound = max(alpha_bound, best_value)
            if alpha_bound >= beta_bound:
                # Beta cutoff! Store as killer move and boost in history
                if move_direction not in KILLER_MOVES[current_ply]:
                    KILLER_MOVES[current_ply][1] = KILLER_MOVES[current_ply][0]
                    KILLER_MOVES[current_ply][0] = move_direction
                MOVE_HISTORY[move_direction] = MOVE_HISTORY.get(move_direction, 0) + current_depth * current_depth
                break
        
        # === STORE IN TRANSPOSITION TABLE ===
        # Flag indicates type of bound: 0=exact, -1=upper bound (fail-low), 1=lower bound (fail-high)
        flag = 0
        if best_value <= alpha_bound:
            flag = -1  # Fail low
        if best_value >= beta_bound:
            flag = 1   # Fail high
        TRANSPOSITION_TABLE[zobrist_hash] = (best_value, current_depth, flag, best_move_found)
        
        return best_value

    # === MINIMIZING PLAYER (OPPONENT'S TURN) ===
    else:
        # Get opponent's legal moves (simpler ordering for speed)
        opponent_moves = []
        for _, delta_row, delta_col in MOVES_FLAT:
            new_row = opp_row + delta_row
            new_col = opp_col + delta_col
            # Check if move is legal
            if 0 <= new_row < board_size and 0 <= new_col < board_size and board[new_row][new_col] == 0:
                # Count opponent's future options from this move
                opponent_mobility = 0
                for _, next_delta_row, next_delta_col in MOVES_FLAT:
                    next_row = new_row + next_delta_row
                    next_col = new_col + next_delta_col
                    if 0 <= next_row < board_size and 0 <= next_col < board_size and board[next_row][next_col] == 0:
                        opponent_mobility += 1
                opponent_moves.append((opponent_mobility, new_row, new_col))
        
        # No legal moves = opponent trapped = I win
        if not opponent_moves:
            return 200000.0 - current_ply  # Positive because I win

        # Sort by mobility (assume opponent plays the best moves)
        opponent_moves.sort(key=lambda x: x[0], reverse=True)

        worst_value = INFINITY
        
        # === SEARCH EACH OPPONENT MOVE ===
        for move_index, (_, new_row, new_col) in enumerate(opponent_moves):
            # === MAKE OPPONENT MOVE ===
            board[new_row][new_col] = 1
            new_zobrist_hash = zobrist_hash ^ ZOBRIST_PLAYER2_POS[opp_row][opp_col] ^ ZOBRIST_PLAYER2_POS[new_row][new_col] ^ ZOBRIST_BOARD[new_row][new_col]
            
            # === LATE MOVE REDUCTION FOR OPPONENT ===
            depth_reduction = 0
            if current_depth >= 3 and move_index >= 3:
                depth_reduction = 1

            # Recursively search my response
            value = minimax(my_row, my_col, new_row, new_col, board, current_depth - 1 - depth_reduction,
                           alpha_bound, beta_bound, True, current_ply + 1, deadline, new_zobrist_hash)
            
            # === LATE MOVE REDUCTION RE-SEARCH ===
            if depth_reduction > 0 and value < beta_bound:
                value = minimax(my_row, my_col, new_row, new_col, board, current_depth - 1,
                               alpha_bound, beta_bound, True, current_ply + 1, deadline, new_zobrist_hash)

            # === UNDO OPPONENT MOVE ===
            board[new_row][new_col] = 0
            
            # === UPDATE WORST VALUE (minimize) ===
            if value < worst_value:
                worst_value = value
            
            # === ALPHA-BETA PRUNING ===
            beta_bound = min(beta_bound, worst_value)
            if beta_bound <= alpha_bound:
                break  # Alpha cutoff!
                
        # === STORE IN TRANSPOSITION TABLE ===
        flag = 0
        if worst_value <= alpha_bound:
            flag = -1
        if worst_value >= beta_bound:
            flag = 1
        TRANSPOSITION_TABLE[zobrist_hash] = (worst_value, current_depth, flag, None)
        return worst_value

def compute_initial_hash(board, my_row, my_col, opp_row, opp_col):
    """
    Compute Zobrist hash for initial board state.
    """
    hash_value = 0
    board_size = len(board)
    
    # XOR in hash for all occupied cells
    for row in range(board_size):
        for col in range(board_size):
            if board[row][col] != 0:
                hash_value ^= ZOBRIST_BOARD[row][col]
    
    # XOR in player positions
    hash_value ^= ZOBRIST_PLAYER1_POS[my_row][my_col]
    hash_value ^= ZOBRIST_PLAYER2_POS[opp_row][opp_col]
    
    return hash_value

def adaptive_depth(board, my_row, my_col, opp_row, opp_col):
    """
    Dynamically adjust search depth based on board state.
    Search deeper in critical moments (late game, close combat).
    """
    # Count empty cells remaining
    open_cells = sum(row.count(0) for row in board)
    
    # Distance to opponent
    manhattan_distance = abs(my_row - opp_row) + abs(my_col - opp_col)
    
    # Start with base depth
    target_depth = BASE_DEPTH
    
    # If board is emptying, endgame is approaching: search deeper
    if open_cells < 200:
        target_depth += 2
    if open_cells < 100:
        target_depth += 4
    
    # If close combat, search deeper to find tactical wins
    if manhattan_distance < 6:
        target_depth += 2
    
    # Cap at maximum allowed depth
    return min(MAX_DEPTH, target_depth)

def root_search(game_state):
    """
    Iterative deepening search from root position.
    Uses Principal Variation Search and Aspiration Windows.
    """
    global node_count
    node_count = 0
    
    # === EXTRACT GAME STATE ===
    my_player = game_state.me
    my_head = my_player.head
    opponent_head = game_state.opponent.head
    board = game_state.board
    board_size = game_state.board_size

    # === TIME MANAGEMENT ===
    deadline = time.time() + TIME_LIMIT - TIME_BUFFER
    
    # === COMPUTE INITIAL ZOBRIST HASH ===
    current_hash = compute_initial_hash(board, my_head.x, my_head.y, opponent_head.x, opponent_head.y)
    
    # === DETERMINE SEARCH DEPTH ===
    max_target_depth = adaptive_depth(board, my_head.x, my_head.y, opponent_head.x, opponent_head.y)

    # === GENERATE LEGAL ROOT MOVES ===
    root_moves = []
    for move_direction, delta_row, delta_col in MOVES_FLAT:
        new_row = my_head.x + delta_row
        new_col = my_head.y + delta_col
        # Check if move is legal
        if 0 <= new_row < board_size and 0 <= new_col < board_size and board[new_row][new_col] == 0:
            root_moves.append((move_direction, new_row, new_col))
    
    # Forced move if only one option
    if not root_moves:
        return [UP]

    best_move = root_moves[0][0]
    best_value = -INFINITY
    
    # === ITERATIVE DEEPENING ===
    # Gradually increase search depth, stopping early if time runs out
    for current_depth in range(1, max_target_depth + 1):
        # Abort if we've run out of time
        if time.time() >= deadline:
            break
        
        # === ASPIRATION WINDOWS ===
        # Use narrow search window around previous best value for efficiency
        alpha_bound = -INFINITY
        beta_bound = INFINITY
        # Only use aspiration if we have a previous result and it's not a mate score
        if current_depth > 3 and abs(best_value) < 50000:
            alpha_bound = best_value - 200
            beta_bound = best_value + 200
        
        current_iteration_best_move = None
        current_iteration_best_value = -INFINITY
        
        # Sort root moves: Principal Variation move first (most likely best)
        root_moves.sort(key=lambda x: 1 if x[0] == best_move else 0, reverse=True)

        # === SEARCH EACH ROOT MOVE ===
        for move_index, (move_direction, new_row, new_col) in enumerate(root_moves):
            # Abort if time expired
            if time.time() >= deadline:
                break
            
            # === MAKE MOVE ===
            board[new_row][new_col] = 1
            new_zobrist_hash = current_hash ^ ZOBRIST_PLAYER1_POS[my_head.x][my_head.y] ^ ZOBRIST_PLAYER1_POS[new_row][new_col] ^ ZOBRIST_BOARD[new_row][new_col]
            
            # === PRINCIPAL VARIATION SEARCH (PVS) AT ROOT ===
            # First move uses full window (most likely to be best)
            if move_index == 0:
                value = minimax(new_row, new_col, opponent_head.x, opponent_head.y, board, current_depth - 1,
                               alpha_bound, beta_bound, False, 1, deadline, new_zobrist_hash)
            else:
                # Other moves use narrow null window first (faster)
                value = minimax(new_row, new_col, opponent_head.x, opponent_head.y, board, current_depth - 1,
                               alpha_bound, alpha_bound + 1, False, 1, deadline, new_zobrist_hash)
                # If narrow window failed high, re-search with full window
                if value > alpha_bound:
                    value = minimax(new_row, new_col, opponent_head.x, opponent_head.y, board, current_depth - 1,
                                   alpha_bound, beta_bound, False, 1, deadline, new_zobrist_hash)
            
            # === UNDO MOVE ===
            board[new_row][new_col] = 0
            
            # === ASPIRATION WINDOW RECOVERY ===
            # If value fell outside window, re-search with infinite window
            if value <= alpha_bound or value >= beta_bound:
                board[new_row][new_col] = 1
                value = minimax(new_row, new_col, opponent_head.x, opponent_head.y, board, current_depth - 1,
                               -INFINITY, INFINITY, False, 1, deadline, new_zobrist_hash)
                board[new_row][new_col] = 0

            # === TIE-BREAKER RANDOMIZATION ===
            # Add small random noise to avoid repeating same moves
            value += random.uniform(-RANDOM_TIE, RANDOM_TIE)
            
            # === UPDATE ITERATION BEST ===
            if value > current_iteration_best_value:
                current_iteration_best_value = value
                current_iteration_best_move = move_direction
            
            # Update alpha bound for next move
            alpha_bound = max(alpha_bound, current_iteration_best_value)

        # === SAVE ITERATION RESULTS ===
        # Only update best move if we completed the full search at this depth
        if time.time() < deadline:
            best_value = current_iteration_best_value
            best_move = current_iteration_best_move

    return [best_move]

def choose_move(game_state: GameState) -> int:
    """
    Main entry point: choose the best move for current game state.
    """
    # Clear transposition table periodically to save memory
    if len(TRANSPOSITION_TABLE) > 500000:
        TRANSPOSITION_TABLE.clear()
    
    # Search for best move
    moves = root_search(game_state)
    return moves[0] if moves else UP

def main():
    """
    Read game states from stdin, compute moves, output to stdout.
    """
    try:
        for line in sys.stdin:
            data = line.strip()
            if not data:
                continue
            # Parse JSON game state
            game_state = GameState.from_json(data)
            # Compute and output move
            move = choose_move(game_state)
            print(move, flush=True)
    except EOFError:
        # End of input
        pass
    except Exception as e:
        # Log errors to stderr
        print(f"Error: {e}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
