# agent_code/ppo_agent/train.py
import os
from typing import List

import events as e
import settings as s
from events import SURVIVED_ROUND

# Try relative import of callbacks - works when package executed as module
try:
    from . import callbacks as ppo_callbacks
except Exception:
    # fallback for other import styles
    import callbacks as ppo_callbacks
from metrics.metrics_tracker import MetricsTracker

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_PATH = "models/ppo_agent.pth"  # consistent save/load location
GAME_REWARDS = {
    "COIN_COLLECTED": 80.0,          # PHASE 2.5: Secondary goal
    "KILLED_OPPONENT": 200.0,        # Not relevant yet (solo training)
    "CRATE_DESTROYED": 50.0,         # REDUCED - survival more important than crates!
    "BOMB_DROPPED": 0.0,             # NEUTRAL - let outcomes determine value
    "BOMB_EXPLODED": 0.0,
    "KILLED_SELF": -800.0,           # MASSIVE PENALTY - death must be avoided!
    "GOT_KILLED": -150.0,            # Not relevant (solo)
    "INVALID_ACTION": -10.0,         # Penalty for invalid actions
    "WAITED": -0.5,                  # Very gentle - don't over-penalize caution
    "SURVIVED_ROUND": 600.0,         # HUGE - survival is PRIMARY GOAL!
    "MOVED_UP": 0.0,                 # No reward for basic movement
    "MOVED_RIGHT": 0.0,
    "MOVED_DOWN": 0.0,
    "MOVED_LEFT": 0.0,
}

# PHASE 2.5 REWARDS: SURVIVAL FIRST, gentle introduction to bombing
STEP_ALIVE_REWARD = 0.7              # VERY HIGH - every step alive is precious
DANGER_PENALTY = -0.5                # VERY GENTLE - allow learning through risks
ESCAPED_DANGER_REWARD = 80.0         # MASSIVE - escaping is critical!
SAFE_BOMB_REWARD = 120.0             # HUGE - successful bombing is jackpot
BOMB_STAY_PENALTY = -40.0            # SEVERE - must move away immediately
UNSAFE_BOMB_PENALTY = -20.0          # Moderate penalty
MOVING_TO_SAFETY_REWARD = 50.0       # MASSIVE - reward every escape step!

# PHASE 2.5 SHAPED REWARDS: Strong escape focus
CRATE_IN_RANGE_REWARD = 10.0         # Moderate - don't over-encourage risky bombs
APPROACHING_COIN_REWARD = 10.0       # Maintain coin collection
SAFE_POSITION_REWARD = 0.5           # Good bonus for safety
EXPLORATION_BONUS = 2.0              # Explore to find opportunities

# --------------------------------------------------------------------------------
# Helper Functions for Reward Shaping
# --------------------------------------------------------------------------------
def count_crates_in_bomb_range(game_state, position):
    """Count how many crates would be destroyed by a bomb at position."""
    if not game_state:
        return 0

    field = game_state["field"]
    x, y = position
    crate_count = 0

    # Check all four directions
    for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
        for step in range(1, s.BOMB_POWER + 1):
            nx, ny = x + dx * step, y + dy * step
            if nx < 0 or ny < 0 or nx >= s.COLS or ny >= s.ROWS:
                break
            if field[nx, ny] == -1:  # Wall
                break
            if field[nx, ny] == 1:  # Crate
                crate_count += 1

    return crate_count


def get_nearest_coin_distance(game_state, position):
    """Get Manhattan distance to nearest coin."""
    if not game_state or not game_state.get("coins"):
        return None

    x, y = position
    coins = game_state["coins"]
    if not coins:
        return None

    min_dist = min(abs(x - cx) + abs(y - cy) for cx, cy in coins)
    return min_dist


def is_approaching_target(old_pos, new_pos, target_positions):
    """Check if agent moved closer to any target."""
    if not target_positions:
        return False

    old_x, old_y = old_pos
    new_x, new_y = new_pos

    for tx, ty in target_positions:
        old_dist = abs(old_x - tx) + abs(old_y - ty)
        new_dist = abs(new_x - tx) + abs(new_y - ty)
        if new_dist < old_dist:
            return True

    return False


# --------------------------------------------------------------------------------
# Setup Training Phase
# --------------------------------------------------------------------------------
def setup_training(self):
    """
    Called once before training starts (not each round).
    Ensures the agent is properly initialized with correct dimensions.
    """
    self.name = "PPO"
    self.logger.info("Setting up PPO training.")
    self.round_counter = getattr(self, "round_counter", 0)

    # Ensure model directory exists
    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir != "":
        os.makedirs(model_dir, exist_ok=True)

    # Check if agent already initialized by callbacks.setup()
    if hasattr(self, "train_agent") and self.train_agent is not None:
        self.logger.info("Using existing train_agent (already initialized by callbacks.setup()).")
    else:
        # Agent not initialized - call setup() to create it
        self.logger.info("train_agent not found. Calling callbacks.setup() to initialize.")
        try:
            ppo_callbacks.setup(self)
            if hasattr(self, "train_agent") and self.train_agent is not None:
                self.logger.info("train_agent successfully initialized via callbacks.setup().")
            else:
                raise RuntimeError("callbacks.setup() did not create train_agent")
        except Exception as exc:
            self.logger.error(f"Failed to initialize train_agent: {exc}")
            raise

    # Log agent configuration
    self.logger.info(f"PPO Agent configured with input_dim={self.train_agent.input_dim}, "
                    f"action_dim={len(ACTIONS)}")
    # Ensure metrics tracker exists
    if not hasattr(self, 'metrics_tracker'):
        
        self.metrics_tracker = MetricsTracker(
            agent_name=self.name,
            save_dir="metrics"
        )
        self.episode_counter = 0
    if not hasattr(self, 'post_bomb_origin'):
        self.post_bomb_origin = None
        self.post_bomb_timer = 0

    # Initialize exploration tracking
    if not hasattr(self, 'visited_positions'):
        self.visited_positions = set()
        self.last_coin_distance = None

    self.logger.info("setup_training completed.")


# --------------------------------------------------------------------------------
# Game Step Callback
# --------------------------------------------------------------------------------
def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """
    Called each time an action was performed.
    Logs transition (s, a, r, s') to PPO buffer.
    """
    if old_game_state is None or new_game_state is None:
        return

    # Safety check
    if not hasattr(self, "train_agent") or self.train_agent is None:
        self.logger.warning("train_agent not available in game_events_occurred.")
        return
    
    if not hasattr(self, "last_obs") or self.last_obs is None:
        self.logger.debug("last_obs not available - skipping transition storage.")
        return

    # =========================================================================
    # START EPISODE ON FIRST STEP - THIS IS THE KEY ADDITION!
    # =========================================================================
    if old_game_state:
        if old_game_state.get('step', 0) == 1:
            # Extract opponent information
            opponent_names = []
            if 'others' in old_game_state and old_game_state['others']:
                for other in old_game_state['others']:
                    if other is not None:
                        # other is typically (name, score, bomb_available, (x, y))
                        opponent_names.append(other[0])
            self.episode_counter=old_game_state.get('round')
            # START THE EPISODE
            self.metrics_tracker.start_episode(
                episode_id=self.episode_counter,
                opponent_types=opponent_names,
                scenario="training"
            )

            # Reset exploration tracking for new episode
            self.visited_positions = set()
            self.last_coin_distance = None

            self.logger.debug(f"Started episode {self.episode_counter} with opponents: {opponent_names}")
    
    
    # --- Reward shaping (customize as needed) ---
    reward = 0.0
    for event in events:
        event_reward = GAME_REWARDS.get(event, 0)
        reward += event_reward
        self.metrics_tracker.record_event(event, reward=event_reward)

    old_danger = None
    new_danger = None
    was_in_danger = False
    if old_game_state and old_game_state.get("self"):
        old_danger = ppo_callbacks.compute_danger_map(old_game_state)
        _, _, _, (ox, oy) = old_game_state["self"]
        was_in_danger = old_danger[ox, oy] >= ppo_callbacks.DANGER_THRESHOLD

    caused_damage = (
        'CRATE_DESTROYED' in events
        or 'KILLED_OPPONENT' in events
        or 'OPPONENT_ELIMINATED' in events
    )
    if 'BOMB_EXPLODED' in events and caused_damage and 'KILLED_SELF' not in events and 'GOT_KILLED' not in events:
        reward += SAFE_BOMB_REWARD
        if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
            self.metrics_tracker.record_event("SAFE_BOMB_EXPLOSION", reward=SAFE_BOMB_REWARD)

    if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
        self.metrics_tracker.record_event("STEP_ALIVE", reward=STEP_ALIVE_REWARD)
    reward += STEP_ALIVE_REWARD

    # NEW: Track escape progress after placing bomb
    if getattr(self, 'post_bomb_timer', 0) > 0 and self_action != 'BOMB':
        if new_game_state and new_game_state.get("self") and old_game_state and old_game_state.get("self"):
            _, _, _, (cx, cy) = new_game_state["self"]
            _, _, _, (ox, oy) = old_game_state["self"]

            # Penalize staying on bomb
            if self.post_bomb_origin and (cx, cy) == self.post_bomb_origin:
                reward += BOMB_STAY_PENALTY
                if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
                    self.metrics_tracker.record_event("STAYED_ON_BOMB", reward=BOMB_STAY_PENALTY)
            else:
                # NEW: Reward moving toward safety after bombing
                old_danger = ppo_callbacks.compute_danger_map(old_game_state)
                new_danger = ppo_callbacks.compute_danger_map(new_game_state)

                # Reward if danger decreased (moving to safety)
                if new_danger[cx, cy] < old_danger[ox, oy]:
                    reward += MOVING_TO_SAFETY_REWARD
                    if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
                        self.metrics_tracker.record_event("MOVING_TO_SAFETY", reward=MOVING_TO_SAFETY_REWARD)

                self.post_bomb_timer = 0
                self.post_bomb_origin = None

        self.post_bomb_timer = max(0, self.post_bomb_timer - 1)
        if self.post_bomb_timer == 0:
            self.post_bomb_origin = None

    if new_game_state:
        new_danger = ppo_callbacks.compute_danger_map(new_game_state)
        if new_game_state.get("self"):
            _, _, _, (nx, ny) = new_game_state["self"]
            if new_danger[nx, ny] >= ppo_callbacks.DANGER_THRESHOLD:
                reward += DANGER_PENALTY
                if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
                    self.metrics_tracker.record_event("ENTERED_DANGER", reward=DANGER_PENALTY)
            if was_in_danger and new_danger[nx, ny] < ppo_callbacks.DANGER_THRESHOLD:
                reward += ESCAPED_DANGER_REWARD
                if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
                    self.metrics_tracker.record_event("ESCAPED_DANGER", reward=ESCAPED_DANGER_REWARD)

    # Shaped reward: Crates in bomb range
    if self_action == 'BOMB' and old_game_state and old_game_state.get("self"):
        _, _, _, (ox, oy) = old_game_state["self"]
        self.post_bomb_origin = (ox, oy)
        self.post_bomb_timer = s.BOMB_TIMER

        # Reward if bomb can hit crates
        crates_in_range = count_crates_in_bomb_range(old_game_state, (ox, oy))
        if crates_in_range > 0:
            crate_reward = CRATE_IN_RANGE_REWARD * crates_in_range
            reward += crate_reward
            if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
                self.metrics_tracker.record_event("CRATES_IN_BOMB_RANGE", reward=crate_reward)

    # Shaped reward: Approaching coins
    if old_game_state and new_game_state:
        if old_game_state.get("self") and new_game_state.get("self"):
            _, _, _, old_pos = old_game_state["self"]
            _, _, _, new_pos = new_game_state["self"]

            coins = new_game_state.get("coins", [])
            if coins and is_approaching_target(old_pos, new_pos, coins):
                reward += APPROACHING_COIN_REWARD
                if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
                    self.metrics_tracker.record_event("APPROACHING_COIN", reward=APPROACHING_COIN_REWARD)

    # Shaped reward: Exploration bonus
    if new_game_state and new_game_state.get("self"):
        _, _, _, (cx, cy) = new_game_state["self"]
        if (cx, cy) not in self.visited_positions:
            reward += EXPLORATION_BONUS
            self.visited_positions.add((cx, cy))
            if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
                self.metrics_tracker.record_event("EXPLORED_NEW_TILE", reward=EXPLORATION_BONUS)

    # Shaped reward: Safe position bonus
    if new_game_state and new_danger is not None:
        if new_game_state.get("self"):
            _, _, _, (nx, ny) = new_game_state["self"]
            if new_danger[nx, ny] < ppo_callbacks.DANGER_THRESHOLD:
                reward += SAFE_POSITION_REWARD
                if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
                    self.metrics_tracker.record_event("SAFE_POSITION", reward=SAFE_POSITION_REWARD)

    # =========================================================================
    # TRACK ACTION
    # =========================================================================
    if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
        # Check if action was actually valid (not just if it's a valid action name)
        # Invalid actions will show up as 'INVALID_ACTION' event from game engine
        action_was_invalid = 'INVALID_ACTION' in events
        self.metrics_tracker.record_action(
            self_action,
            is_valid=not action_was_invalid
        )
        
    # Not terminal yet
    done = False

    # Store transition
    try:
        self.train_agent.store_transition(
            obs=self.last_obs,
            action=self.last_action,
            log_prob=self.last_log_prob,
            reward=reward,
            value=self.last_value,
            done=done,
            mask=getattr(self, "last_action_mask", None)
        )
    except Exception as exc:
        self.logger.error(f"Failed to store transition: {exc}")


# --------------------------------------------------------------------------------
# End of Round Callback
# --------------------------------------------------------------------------------
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called once at the end of each round.
    Finalizes reward, updates PPO, and saves model checkpoint.
    """
    # Safety check
    if not hasattr(self, "train_agent") or self.train_agent is None:
        self.logger.warning("train_agent not available at end_of_round - skipping update.")
        return
    
    # Final reward computation
    # NOTE: Most events are already recorded in game_events_occurred()
    # EXCEPT for death events (KILLED_SELF, GOT_KILLED) which only occur when agent is dead
    # and game_events_occurred() is NOT called for dead agents (see environment.py line 408)
    # So we must record those events here
    final_reward = 0.0
    for event in events:
        event_reward = GAME_REWARDS.get(event, 0)
        final_reward += event_reward

        # Record events that only happen at end of round (not during gameplay)
        # These are: KILLED_SELF, GOT_KILLED, SURVIVED_ROUND, OPPONENT_ELIMINATED
        if event in ['KILLED_SELF', 'GOT_KILLED', 'SURVIVED_ROUND', 'OPPONENT_ELIMINATED']:
            if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
                self.metrics_tracker.record_event(event, reward=event_reward)

    # Store final transition if we have last_obs
    if hasattr(self, "last_obs") and self.last_obs is not None:
        try:
            # This is a terminal state
            self.train_agent.store_transition(
                obs=self.last_obs,
                action=self.last_action,
                log_prob=self.last_log_prob,
                reward=final_reward,
                value=self.last_value,
                done=True,  # Terminal state
                mask=getattr(self, "last_action_mask", None)
            )
        except Exception as exc:
            self.logger.error(f"Failed to store final transition: {exc}")

    # Perform PPO update
    try:
        update_info = self.train_agent.update()
        if isinstance(update_info, dict):
            if update_info.get("updated"):
                self.logger.info(
                    "PPO update completed: steps=%d actor_loss=%.4f critic_loss=%.4f entropy=%.4f kl=%.4f",
                    update_info.get("num_samples", 0),
                    update_info.get("avg_actor_loss", 0.0),
                    update_info.get("avg_critic_loss", 0.0),
                    update_info.get("avg_entropy", 0.0),
                    update_info.get("avg_kl", 0.0),
                )
            else:
                self.logger.debug(
                    "Skipping PPO update (buffer=%d / required=%d)",
                    update_info.get("num_samples", 0),
                    getattr(self.train_agent, "batch_size", 0),
                )
        else:
            self.logger.info("PPO update completed successfully.")
    except Exception as exc:
        self.logger.error(f"PPO update failed: {exc}")

    # Save model checkpoint
    try:
        self.train_agent.save(MODEL_PATH)
        self.logger.info(f"Model checkpoint saved to '{MODEL_PATH}'")
    except Exception as exc:
        self.logger.error(f"Failed to save PPO model: {exc}")

    # Increment round counter
    self.round_counter = getattr(self, "round_counter", 0) + 1
    self.logger.info(f"Round {self.round_counter} ended.")
    
    won = False
    rank = 4

    if e.SURVIVED_ROUND in events:
        if 'others' in last_game_state:
            alive_opponents = sum(1 for o in last_game_state['others'] if o)
            won = (alive_opponents == 0)
            rank = 1 if won else 2
    current_step = last_game_state.get('step')
    self.metrics_tracker.end_episode(
        won="WON" in events,
        rank=rank,
        survival_steps=current_step,
        total_steps=400
    )
    self.metrics_tracker.save()
    
    # Log training statistics if available
    if len(self.train_agent.memory["rewards"]) > 0:
        total_reward = sum(self.train_agent.memory["rewards"])
        self.logger.info(f"Round {self.round_counter} - Total reward: {total_reward:.2f}")
