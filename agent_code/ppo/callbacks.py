# callbacks.py - WITH GAMEPLAY METRICS TRACKING

import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from metrics.metrics_tracker import MetricsTracker

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
MODEL_PATH = "models/ppo_agent.pth"


# ---------------------------------------------------------------------
# PPO Network (unchanged)
# ---------------------------------------------------------------------
class PPONetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(PPONetwork, self).__init__()
        self.input_dim = input_dim
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        probs = self.actor(x)
        value = self.critic(x)
        return probs, value


# ---------------------------------------------------------------------
# PPO Agent (unchanged - keeping your exact implementation)
# ---------------------------------------------------------------------
class PPOAgent:
    def __init__(self, input_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2, update_epochs=4):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.input_dim = input_dim
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
            
        self.model = PPONetwork(input_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.memory = {
            "obs": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "dones": []
        }

    def featurize(self, game_state: dict) -> np.ndarray:
        EXPECTED_FEATURE_SIZE = 291
        
        if game_state is None:
            return np.zeros(EXPECTED_FEATURE_SIZE, dtype=np.float32)

        try:
            board = np.array(game_state["field"], dtype=np.float32).flatten()
            position = np.array(game_state["self"][3], dtype=np.float32)
            features = np.concatenate([board, position])
            
            if features.shape[0] != EXPECTED_FEATURE_SIZE:
                if features.shape[0] < EXPECTED_FEATURE_SIZE:
                    features = np.pad(features, (0, EXPECTED_FEATURE_SIZE - features.shape[0]))
                else:
                    features = features[:EXPECTED_FEATURE_SIZE]
            
            return features
            
        except Exception as e:
            print(f"Error in featurize: {e}")
            return np.zeros(EXPECTED_FEATURE_SIZE, dtype=np.float32)

    def select_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        if obs_tensor.shape[1] != self.input_dim:
            raise ValueError(f"Observation dimension {obs_tensor.shape[1]} doesn't match model input_dim {self.input_dim}")
        
        probs, value = self.model(obs_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return int(action.item()), float(log_prob.item()), float(value.item())

    def store_transition(self, obs, action, log_prob, reward, value, done):
        self.memory["obs"].append(obs)
        self.memory["actions"].append(action)
        self.memory["log_probs"].append(log_prob)
        self.memory["rewards"].append(reward)
        self.memory["values"].append(value)
        self.memory["dones"].append(done)

    def compute_advantages(self, next_value=0):
        rewards = np.array(self.memory["rewards"], dtype=np.float32)
        values = np.array(self.memory["values"] + [next_value], dtype=np.float32)
        dones = np.array(self.memory["dones"], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    def update(self):
        if len(self.memory["obs"]) == 0:
            return

        obs = torch.tensor(np.vstack(self.memory["obs"]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.memory["actions"], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(self.memory["log_probs"], dtype=torch.float32).to(self.device)

        advantages, returns = self.compute_advantages()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        for _ in range(self.update_epochs):
            probs, values = self.model(obs)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_tensor

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
            loss = actor_loss + critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        for k in self.memory.keys():
            self.memory[k] = []

    def save(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
        }
        torch.save(checkpoint, path)

    def load(self, path=MODEL_PATH):
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return False
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                saved_input_dim = checkpoint.get('input_dim', None)
                
                if saved_input_dim is not None and saved_input_dim != self.input_dim:
                    print(f"Warning: Saved model input_dim ({saved_input_dim}) != current input_dim ({self.input_dim})")
                    print("Cannot load model with mismatched dimensions. Using random initialization.")
                    return False
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            print(f"Successfully loaded model from {path}")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False


# ---------------------------------------------------------------------
# Bomberman Callbacks — WITH GAMEPLAY METRICS
# ---------------------------------------------------------------------

def setup(self):
    """Called once before the first game starts."""
    FEATURE_DIM = 291
    self.name = "PPO"
    self.train_agent = PPOAgent(input_dim=FEATURE_DIM, action_dim=len(ACTIONS))
    
    # Load model
    if os.path.exists(MODEL_PATH):
        success = self.train_agent.load(MODEL_PATH)
        if success:
            self.logger.info(f"Loaded pretrained PPO model from {MODEL_PATH}")
        else:
            self.logger.info(f"Starting with fresh PPO model (input_dim={FEATURE_DIM})")
    else:
        self.logger.info(f"No pretrained model found. Initialized new PPOAgent with input_dim={FEATURE_DIM}")
    
    # Initialize state tracking
    self.last_obs = None
    self.last_action = None
    self.last_log_prob = None
    self.last_value = None
    
    # Initialize metrics tracker
    self.metrics_tracker = MetricsTracker(
        agent_name=self.name,
        save_dir="metrics"
    )
    self.episode_counter = 0
    
    # Track if episode is active
    self.episode_active = False
    self.current_step = 0
    
    self.logger.info("PPO callbacks.setup() finished with metrics tracking.")


def act(self, game_state, events=[]):
    """Called each game step to select an action."""
    
    # Safety check
    if not hasattr(self, 'train_agent') or self.train_agent is None:
        self.logger.error("train_agent not initialized! Calling setup()...")
        setup(self)
    
    # Initialize metrics tracker if missing
    if not hasattr(self, 'metrics_tracker'):
        self.metrics_tracker = MetricsTracker(
            agent_name=self.name,
            save_dir="metrics"
        )
        self.episode_counter = 0
        self.episode_active = False
        self.current_step = 0
    
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
    # =========================================================================
    # SELECT ACTION
    # =========================================================================
    obs = self.train_agent.featurize(game_state)
    
    try:
        action_idx, log_prob, value = self.train_agent.select_action(obs)
    except Exception as e:
        self.logger.error(f"Error selecting action: {e}")
        action_idx = np.random.randint(0, len(ACTIONS))
        log_prob = 0.0
        value = 0.0
    
    # Store for potential training use
    self.last_obs = obs
    self.last_action = action_idx
    self.last_log_prob = log_prob
    self.last_value = value
    
    action = ACTIONS[action_idx]
    
    # =========================================================================
    # TRACK ACTION
    # =========================================================================
    if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
        self.metrics_tracker.record_action(action, is_valid=True)
    
    return action


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
    # If episode is still active, end it now
    if hasattr(self, 'episode_active') and self.episode_active:
        _end_gameplay_episode(self, last_game_state, events, died=False)
    
    # Log final events if you want to track them
    if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
        # This shouldn't happen since we ended the episode above,
        # but just in case there's a timing issue
        self.logger.warning("Episode still active in end_of_round - ending now")
        _end_gameplay_episode(self, last_game_state, events, died=False)


# =============================================================================
# OPTIONAL: Add event tracking during gameplay
# =============================================================================

def game_events_occurred(self, old_game_state: dict, self_action: str, 
                        new_game_state: dict, events: list):
    """
    Optional: Track events during gameplay (not just training).
    Only available if you're running with --scenario or certain modes.
    
    Note: This callback may not be called during regular gameplay,
    only during training. Check your framework documentation.
    """
    if not hasattr(self, 'metrics_tracker'):
        return
    
    if not self.metrics_tracker.current_episode:
        return
    
    # Event rewards (customize as needed)
    EVENT_REWARDS = {
        'COIN_COLLECTED': 10,
        'KILLED_OPPONENT': 50,
        'KILLED_SELF': -100,
        'GOT_KILLED': -50,
        'CRATE_DESTROYED': 5,
        'INVALID_ACTION': -5,
    }
    
    # Track events
    for event in events:
        reward = EVENT_REWARDS.get(event, 0)
        self.metrics_tracker.record_event(event, reward=reward)