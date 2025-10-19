import os
import pickle
import random

import numpy as np
from metrics.metrics_tracker import MetricsTracker

# ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup Q-table and hyperparameters.
    Loads saved model if available for continued training.
    """
    model_path = "my-saved-model.pt"
    self.name = "Q Learning"
    if os.path.isfile(model_path):
        print("Loading existing model for continued training...")
        with open(model_path, "rb") as file:
            self.model = pickle.load(file)
    else:
        print("No existing model found. Starting new Q-table.")
        self.model = {}

    self.metrics_tracker = MetricsTracker(
        agent_name=self.name,
        save_dir="metrics"
    )
    self.episode_counter = 0
    # Track if episode is active
    self.episode_active = False
    self.current_step = 0
    
    self.epsilon = getattr(self, "epsilon", 1.0)     # exploration rate
    
    # Hyperparameters
    # self.epsilon = max(0.05, self.epsilon * 0.995)  # exploration rate
    self.eplison = 0.1
    self.alpha = 0.05  # learning rate
    self.gamma = 0.95  # discount factor


def act(self, game_state, events=[]) -> str:
    """
    Choose action using epsilon-greedy policy.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_features(game_state)

    # Epsilon-greedy exploration
    if self.train and random.random() < self.epsilon:
        self.logger.debug("Exploring: choosing random action")
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .15, .05])  # biased random choice

    # Exploitation: choose best action based on Q-values
    if features not in self.model:
        # Initialize Q-values for unseen state
        self.model[features] = {action: 0.0 for action in ACTIONS}

    q_values = self.model[features]
    best_action = max(q_values, key=q_values.get)

    self.logger.debug(f"Exploiting: choosing action {best_action}")
    
    if not hasattr(self, 'metrics_tracker'):
        self.metrics_tracker = MetricsTracker(
            agent_name=self.name,
            save_dir="metrics"
        )
        self.episode_counter = 0
    
    # =========================================================================
    # START EPISODE ON FIRST STEP
    # =========================================================================
    if game_state and game_state.get('step', 0) == 1:
        # Extract opponent information
        opponent_names = []
        if 'others' in game_state and game_state['others']:
            for other in game_state['others']:
                if other is not None:
                    opponent_names.append(other[0])
        
        # Get episode ID from game state
        self.episode_counter = game_state.get('round', self.episode_counter)
        
        # START THE EPISODE
        self.metrics_tracker.start_episode(
            episode_id=self.episode_counter,
            opponent_types=opponent_names,
            map_name="default",
            scenario="gameplay"
        )
        
        self.episode_active = True
        self.current_step = 0
        self.logger.debug(f"Started gameplay episode {self.episode_counter}")
    
    # =========================================================================
    # CHECK FOR EPISODE END (AGENT ELIMINATED OR GAME OVER)
    # =========================================================================
    if game_state and self.episode_active:
        # Check if this agent is dead/eliminated
        agent_alive = True
        
        # Method 1: Check if 'self' exists in game_state
        if 'self' not in game_state or game_state['self'] is None:
            agent_alive = False
        
        # Method 2: Check for explicit dead flag (if your framework has it)
        if 'dead' in game_state and game_state.get('dead', False):
            agent_alive = False
        
        # Method 3: Check game state for end conditions
        # The game might set 'round_finished' or similar
        if game_state.get('round_finished', False):
            agent_alive = False  # Treat as end
        
        # If agent died, end the episode
        if not agent_alive and self.episode_active:
            _end_gameplay_episode(self, game_state, died=True)
    
    # Increment step counter
    if game_state:
        self.current_step = game_state.get('step', self.current_step + 1)
    
    for event in events:
        self.metrics_tracker.record_event(event)    
    if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
        self.metrics_tracker.record_action(best_action, is_valid=True)
    return best_action

def _end_gameplay_episode(self, game_state, events, died=False):
    """
    Helper function to end episode during gameplay.
    
    Args:
        game_state: Final game state
        died: Whether agent died/was eliminated
    """
    if not hasattr(self, 'metrics_tracker') or not self.episode_active:
        return
    
    if not self.metrics_tracker.current_episode:
        return
    
    for event in events:
        self.metrics_tracker.record_event(event)
        
    # Determine outcome
    won = False
    rank = 4  # Default to last place
    
    if not died:
        # Agent survived - check if won
        if 'others' in game_state and game_state['others']:
            alive_opponents = sum(1 for o in game_state['others'] if o is not None)
            won = (alive_opponents == 0)
            rank = 1 if won else 2
        else:
            # No opponent info, assume won if survived
            won = True
            rank = 1
    else:
        # Agent died
        won = False
        if 'others' in game_state and game_state['others']:
            alive_opponents = sum(1 for o in game_state['others'] if o is not None)
            rank = alive_opponents + 2  # Finished below all alive opponents
        else:
            rank = 4
    
    survival_steps = self.current_step
    total_steps = game_state.get('step', survival_steps) if game_state else survival_steps
    
    # End episode
    self.metrics_tracker.end_episode(
        won=won,
        rank=rank,
        survival_steps=survival_steps,
        total_steps=total_steps,
        metadata={
            'mode': 'gameplay',
            'died': died,
            'episode': self.episode_counter
        }
    )
    
    self.episode_active = False
    self.episode_counter += 1
    
    current_step = game_state.get('step')
    self.metrics_tracker.end_episode(
        won="WON" in events,
        rank=rank,
        survival_steps=current_step,
        total_steps=400
    )
    self.metrics_tracker.save()
    print(f"Ended gameplay episode: won={won}, rank={rank}, steps={survival_steps}")


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """
    Called at the end of each round during gameplay.
    This is ONLY called in play mode, not during training.
    """
    print("End of round callback triggered.")
    # If episode is still active, end it now
    if hasattr(self, 'episode_active') and self.episode_active:
        _end_gameplay_episode(self, last_game_state, events, died=False)
    
    # Log final events if you want to track them
    if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
        # This shouldn't happen since we ended the episode above,
        # but just in case there's a timing issue
        self.logger.warning("Episode still active in end_of_round - ending now")
        _end_gameplay_episode(self, last_game_state, events, died=False)

def state_to_features(game_state: dict) -> tuple:
    """
    Convert game state to feature tuple for Q-table lookup.

    Features extracted:
    - What's in each adjacent tile (up, right, down, left)
    - Direction to nearest coin
    - Whether bomb is available
    - Basic danger assessment

    :param game_state: A dictionary describing the current game board.
    :return: tuple of features (hashable for Q-table)
    """
    if game_state is None:
        return None

    # Extract basic info
    _, score, bomb_available, (x, y) = game_state['self']
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']

    features = []

    # Feature 1-4: Adjacent tiles (UP, RIGHT, DOWN, LEFT)
    adjacent_positions = [
        (x, y - 1),  # UP
        (x + 1, y),  # RIGHT
        (x, y + 1),  # DOWN
        (x - 1, y),  # LEFT
    ]

    for pos_x, pos_y in adjacent_positions:
        tile_feature = get_tile_feature(pos_x, pos_y, field, bombs, explosion_map)
        features.append(tile_feature)

    # Feature 5: Direction to nearest coin
    if len(coins) > 0:
        coin_direction = get_direction_to_nearest(x, y, coins)
        features.append(coin_direction)
    else:
        features.append('no_coin')

    # Feature 6: Can place bomb?
    features.append('can_bomb' if bomb_available else 'no_bomb')

    # Feature 7: Am I in danger? (standing on explosion or near bomb)
    in_danger = is_in_danger(x, y, bombs, explosion_map)
    features.append('danger' if in_danger else 'safe')

    return tuple(features)


def get_tile_feature(x, y, field, bombs, explosion_map):
    """
    Determine what's at a specific tile.
    Returns: 'wall', 'crate', 'bomb', 'explosion', 'free'
    """
    # Check if out of bounds or wall
    if x < 0 or x >= field.shape[0] or y < 0 or y >= field.shape[1]:
        return 'wall'

    if field[x, y] == -1:
        return 'wall'

    if field[x, y] == 1:
        return 'crate'

    # Check for active explosion
    if explosion_map[x, y] > 0:
        return 'explosion'

    # Check for bomb
    for bomb_pos, _ in bombs:
        if bomb_pos == (x, y):
            return 'bomb'

    return 'free'


def get_direction_to_nearest(x, y, targets):
    """
    Get general direction to nearest target (coin, crate, etc.).
    Returns: 'up', 'right', 'down', 'left', 'here'
    """
    if len(targets) == 0:
        return 'here'

    # Find nearest target using Manhattan distance
    min_distance = float('inf')
    nearest = None

    for target in targets:
        if isinstance(target, tuple):
            tx, ty = target
        else:
            tx, ty = target  # assuming target is array-like

        distance = abs(tx - x) + abs(ty - y)
        if distance < min_distance:
            min_distance = distance
            nearest = (tx, ty)

    if nearest is None:
        return 'here'

    tx, ty = nearest

    # Determine primary direction
    dx = tx - x
    dy = ty - y

    # Prioritize horizontal or vertical based on larger difference
    if abs(dx) > abs(dy):
        return 'right' if dx > 0 else 'left'
    elif abs(dy) > abs(dx):
        return 'down' if dy > 0 else 'up'
    elif dx > 0:
        return 'right'
    elif dx < 0:
        return 'left'
    elif dy > 0:
        return 'down'
    elif dy < 0:
        return 'up'
    else:
        return 'here'


def is_in_danger(x, y, bombs, explosion_map):
    """
    Check if current position is dangerous.
    Returns True if standing on explosion or in bomb blast radius.
    """
    # Currently in explosion
    if explosion_map[x, y] > 0:
        return True

    # Check if in blast radius of any bomb
    for (bx, by), timer in bombs:
        # Bombs have blast radius of 3 in each direction
        if bx == x and abs(by - y) <= 3:
            return True
        if by == y and abs(bx - x) <= 3:
            return True

    return False