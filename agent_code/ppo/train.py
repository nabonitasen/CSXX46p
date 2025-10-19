# agent_code/ppo_agent/train.py
import os
from typing import List

# Try relative import of callbacks - works when package executed as module
try:
    from . import callbacks as ppo_callbacks
except Exception:
    # fallback for other import styles
    import callbacks as ppo_callbacks
from metrics.metrics_tracker import MetricsTracker

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_PATH = "models/ppo_agent.pth"  # consistent save/load location
GAME_REWARDS ={
        "COIN_COLLECTED": 10.0,
        "KILLED_OPPONENT": 50.0,
        "CRATE_DESTROYED": 5.0,
        "BOMB_DROPPED": 1.0,
        "KILLED_SELF": -100.0,
        "GOT_KILLED": -50.0,
        "INVALID_ACTION": -5.0,
        "WAITED": -1.0,
        "SURVIVED_ROUND": 30.0,
        
    }

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
        except Exception as e:
            self.logger.error(f"Failed to initialize train_agent: {e}")
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
            
            self.logger.debug(f"Started episode {self.episode_counter} with opponents: {opponent_names}")
    
    
    # --- Reward shaping (customize as needed) ---
    reward = 0.0
    for event in events:
        event_reward = GAME_REWARDS.get(event, 0)
        reward += event_reward
        self.metrics_tracker.record_event(event, reward=event_reward)

    # =========================================================================
    # TRACK ACTION
    # =========================================================================
    if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
        self.metrics_tracker.record_action(self.last_action, is_valid=True)
        
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
            done=done
        )
    except Exception as e:
        self.logger.error(f"Failed to store transition: {e}")


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
    final_reward = 0.0
    for event in events:
        event_reward = GAME_REWARDS.get(event, 0)
        final_reward += event_reward
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
                done=True  # Terminal state
            )
        except Exception as e:
            self.logger.error(f"Failed to store final transition: {e}")

    # Perform PPO update
    try:
        self.train_agent.update()
        self.logger.info("PPO update completed successfully.")
    except Exception as e:
        self.logger.error(f"PPO update failed: {e}")

    # Save model checkpoint
    try:
        self.train_agent.save(MODEL_PATH)
        self.logger.info(f"Model checkpoint saved to '{MODEL_PATH}'")
    except Exception as e:
        self.logger.error(f"Failed to save PPO model: {e}")

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