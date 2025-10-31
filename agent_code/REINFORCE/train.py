import pickle
import numpy as np
from typing import List
import torch
import torch.optim as optim

import events as e
from .callbacks import state_to_features, device
from collections import deque

def setup_training(self):
    """
    Initialize training-specific variables for REINFORCE.

    REINFORCE is a Monte Carlo policy gradient method:
    - Collect full episode (all states, actions, rewards)
    - Calculate returns (sum of future rewards)
    - Update policy to increase probability of good actions

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.info("Setting up REINFORCE training mode.")

    # Episode storage (cleared after each episode)
    self.saved_log_probs = []  # Log probabilities of actions
    self.rewards = []  # Rewards at each step

    # Training hyperparameters
    self.gamma = 0.99  # Discount factor

    # Statistics tracking
    self.episode_rewards = []


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """
    Called after each step.

    For REINFORCE: Just collect data, don't update yet!

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state before action
    :param self_action: The action taken
    :param new_game_state: The state after action
    :param events: Events that occurred
    """
    self.logger.debug(f'Events: {", ".join(events)}')

    # Calculate reward
    reward = reward_from_events(events)

    # Store the log_prob (set in callbacks.act()) and reward
    if hasattr(self, 'last_log_prob'):
        self.saved_log_probs.append(self.last_log_prob)
        self.rewards.append(reward)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at end of episode.

    THIS IS WHERE WE UPDATE THE POLICY!

    REINFORCE algorithm:
    1. Calculate returns: G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
    2. Calculate loss: -Σ log π(a_t|s_t) * G_t
    3. Backpropagate and update policy

    :param self: Agent object
    :param last_game_state: Final state
    :param last_action: Final action
    :param events: Final events
    """
    self.logger.debug(f'Final events: {", ".join(events)}')

    # Get final reward
    final_reward = reward_from_events(events)

    # Store final transition
    if hasattr(self, 'last_log_prob'):
        self.saved_log_probs.append(self.last_log_prob)
        self.rewards.append(final_reward)

    # Calculate total episode reward (for logging)
    total_reward = sum(self.rewards)
    self.episode_rewards.append(total_reward)

    # ============================================
    # REINFORCE UPDATE (the main algorithm!)
    # ============================================

    if len(self.saved_log_probs) > 0:
        # Step 1: Calculate returns (discounted cumulative rewards)
        returns = deque()
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G  # G_t = r_t + γ*G_{t+1}
            returns.appendleft(G)

        # Convert to tensor
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # Normalize returns (helps training stability)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Step 2: Calculate policy loss
        policy_loss = []
        for log_prob, G in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * G)  # -log π(a|s) * G

        policy_loss = torch.stack(policy_loss).sum()

        # Step 3: Backpropagate
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.logger.info(f'Policy loss: {policy_loss.item():.4f}')

    # ============================================
    # Logging and cleanup
    # ============================================

    self.logger.info(f'Round {last_game_state["round"]} finished')
    self.logger.info(f'Episode reward: {total_reward:.2f}')
    self.logger.info(f'Episode length: {len(self.rewards)} steps')

    # Log average over last 100 episodes
    if len(self.episode_rewards) >= 100:
        avg_reward = np.mean(self.episode_rewards[-100:])
        self.logger.info(f'Avg reward (last 100): {avg_reward:.2f}')

    # Clear episode data for next round
    self.saved_log_probs = []
    self.rewards = []
    # Delete the reference to the last log_prob tensor to prevent it from
    # being used in the next episode after its graph has been freed.
    if hasattr(self, 'last_log_prob'):
        del self.last_log_prob

    # Save model every 100 rounds
    if last_game_state["round"] % 100 == 0:
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, "my-saved-model.pth")
        self.logger.info("Model saved.")


def reward_from_events(events: List[str]) -> float:
    """
    Simple reward function.

    :param events: List of game events
    :return: Total reward
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -50,
        e.GOT_KILLED: -30,
        e.INVALID_ACTION: -2,
        e.WAITED: -0.1,
        e.CRATE_DESTROYED: 2,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    return reward_sum
