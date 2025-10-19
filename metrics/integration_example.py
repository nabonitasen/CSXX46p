"""
Integration Example: How to use the metrics system with your agents

This file demonstrates how to integrate the metrics tracking system with
your PPO agent and other RL/LLM agents for comprehensive evaluation.
"""

from metrics_tracker import MetricsTracker, EpisodeMetrics
from metrics_comparison import AgentComparator, PerformanceAnalyzer, load_and_compare_agents
from metrics_visualization import MetricsVisualizer, generate_all_plots


# =============================================================================
# INTEGRATION WITH CALLBACKS.PY AND TRAIN.PY
# =============================================================================

def integrate_with_callbacks():
    """
    Example of how to integrate MetricsTracker with your callbacks.py
    
    Add this to your callbacks.py setup() function:
    """
    code_example = """
# In callbacks.py setup() function:

from metrics_tracker import MetricsTracker

def setup(self):
    # Existing setup code...
    self.train_agent = PPOAgent(...)
    
    # Add metrics tracker
    self.metrics_tracker = MetricsTracker(
        agent_name="PPO_Agent",
        save_dir="metrics"
    )
    
    self.current_episode_id = 0
    """
    
    print("=== INTEGRATION WITH callbacks.py ===")
    print(code_example)


def integrate_with_act():
    """
    Example of how to track actions in act() function
    """
    code_example = """
# In callbacks.py act() function:

def act(self, game_state):
    # Existing action selection...
    obs = self.train_agent.featurize(game_state)
    action_idx, log_prob, value = self.train_agent.select_action(obs)
    action = ACTIONS[action_idx]
    
    # Track the action
    if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
        self.metrics_tracker.record_action(action, is_valid=True)
    
    return action
    """
    
    print("\n=== INTEGRATION WITH act() ===")
    print(code_example)


def integrate_with_training():
    """
    Example of how to integrate with train.py
    """
    code_example = """
# In train.py

from metrics_tracker import MetricsTracker

def setup_training(self):
    # Existing setup...
    
    # Initialize metrics tracker if not already done
    if not hasattr(self, 'metrics_tracker'):
        self.metrics_tracker = MetricsTracker(
            agent_name="PPO_Agent",
            save_dir="metrics"
        )
    
    self.current_episode_id = 0


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    # Existing code...
    
    # Track events and compute reward
    reward = 0
    for event in events:
        if event == "COIN_COLLECTED":
            reward += 10
            self.metrics_tracker.record_event(event, reward=10)
        elif event == "KILLED_OPPONENT":
            reward += 50
            self.metrics_tracker.record_event(event, reward=50)
        # ... etc for all events
    
    # Existing PPO storage code...


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    # Determine outcome
    won = "WON" in events or self.game_state.get('self_won', False)
    rank = self.game_state.get('self_rank', 1)
    survival_steps = self.game_state.get('step', 0)
    total_steps = self.game_state.get('max_steps', 400)
    
    # End episode tracking
    if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
        self.metrics_tracker.end_episode(
            won=won,
            rank=rank,
            survival_steps=survival_steps,
            total_steps=total_steps,
            metadata={'round': self.round_counter}
        )
    
    # Existing PPO update code...
    
    # Save metrics periodically
    if self.round_counter % 100 == 0:
        self.metrics_tracker.save()
        print(f"Metrics saved at round {self.round_counter}")
    """
    
    print("\n=== INTEGRATION WITH train.py ===")
    print(code_example)


# =============================================================================
# COMPLETE INTEGRATION EXAMPLE
# =============================================================================

class BombermanAgentWrapper:
    """
    Example wrapper that integrates metrics tracking with any agent.
    
    This wrapper can be used with PPO, DQN, LLM, or any other agent type.
    """
    
    def __init__(self, agent, agent_name: str):
        """
        Initialize wrapper.
        
        Args:
            agent: The actual agent (PPO, DQN, LLM, etc.)
            agent_name: Name for metrics tracking
        """
        self.agent = agent
        self.metrics_tracker = MetricsTracker(
            agent_name=agent_name,
            save_dir="metrics"
        )
        self.episode_id = 0
        self.current_step = 0
    
    def start_episode(self, opponent_types=None, map_name="default"):
        """Start a new episode."""
        self.episode_id += 1
        self.current_step = 0
        
        self.metrics_tracker.start_episode(
            episode_id=self.episode_id,
            opponent_types=opponent_types or [],
            map_name=map_name
        )
    
    def act(self, game_state):
        """Select action and track it."""
        # Get action from underlying agent
        action = self.agent.act(game_state)
        
        # Track the action
        self.metrics_tracker.record_action(action, is_valid=True)
        self.current_step += 1
        
        return action
    
    def observe(self, events, reward):
        """
        Observe events and track them.
        
        Args:
            events: List of event names
            reward: Total reward received
        """
        # Track each event
        for event in events:
            self.metrics_tracker.record_event(event, reward=reward)
    
    def end_episode(self, won, rank, total_steps):
        """End episode and finalize metrics."""
        self.metrics_tracker.end_episode(
            won=won,
            rank=rank,
            survival_steps=self.current_step,
            total_steps=total_steps
        )
    
    def save_metrics(self):
        """Save metrics to disk."""
        self.metrics_tracker.save()
    
    def get_summary(self):
        """Get summary statistics."""
        return self.metrics_tracker.get_summary_stats()


# =============================================================================
# EVALUATION SCRIPT EXAMPLE
# =============================================================================

def evaluate_single_agent_example():
    """
    Example: Evaluate a single agent and generate plots.
    """
    print("\n=== SINGLE AGENT EVALUATION ===\n")
    
    code_example = """
from metrics_tracker import MetricsTracker
from metrics_visualization import MetricsVisualizer

# Load saved metrics
tracker = MetricsTracker(agent_name="PPO_Agent")
tracker.load("metrics/PPO_Agent_20241019_120000.pkl")

# Get summary statistics
summary = tracker.get_summary_stats()
print("Summary Statistics:")
for key, value in summary.items():
    print(f"  {key}: {value}")

# Get learning curve data
episodes, rewards, moving_avg = tracker.get_learning_curve(
    metric='total_reward',
    window_size=10
)

# Compare performance by opponent
opp_comparison = tracker.compare_by_opponent()
print("\\nPerformance by Opponent:")
for opp, stats in opp_comparison.items():
    print(f"  {opp}: {stats['win_rate']:.3f} win rate")

# Generate visualizations
visualizer = MetricsVisualizer(save_dir="plots")
visualizer.plot_learning_curves([tracker], metric='total_reward')
visualizer.plot_performance_vs_opponents(tracker)
    """
    
    print(code_example)


def compare_multiple_agents_example():
    """
    Example: Compare multiple agents.
    """
    print("\n=== MULTIPLE AGENT COMPARISON ===\n")
    
    code_example = """
from metrics_tracker import MetricsTracker
from metrics_comparison import AgentComparator, load_and_compare_agents
from metrics_visualization import MetricsVisualizer, generate_all_plots

# Load multiple agents
tracker1 = MetricsTracker(agent_name="PPO_Agent")
tracker1.load("metrics/PPO_Agent_20241019_120000.pkl")

tracker2 = MetricsTracker(agent_name="DQN_Agent")
tracker2.load("metrics/DQN_Agent_20241019_120000.pkl")

tracker3 = MetricsTracker(agent_name="LLM_Agent")
tracker3.load("metrics/LLM_Agent_20241019_120000.pkl")

trackers = [tracker1, tracker2, tracker3]

# Create comparator
comparator = AgentComparator(trackers)

# Generate comparison report
report = comparator.generate_comparison_report(
    output_file="comparison_results/report.txt"
)
print(report)

# Export comparison data
comparator.export_comparison_json("comparison_results/data.json")

# Statistical comparisons
pairwise = comparator.pairwise_comparison(
    metric='total_reward',
    test='mannwhitney'
)

for pair, results in pairwise.items():
    if results['significant_at_0.05']:
        print(f"{pair}: Significant difference (p={results['p_value']:.4f})")

# Sample efficiency
efficiency = comparator.compute_sample_efficiency(
    target_performance=0.5,
    metric='win_rate'
)
print("\\nSample Efficiency (episodes to 50% win rate):")
for agent, episodes in efficiency.items():
    if episodes:
        print(f"  {agent}: {episodes} episodes")

# Generate all plots
generate_all_plots(trackers, comparator, save_dir="plots")
    """
    
    print(code_example)


def advanced_analysis_example():
    """
    Example: Advanced performance analysis.
    """
    print("\n=== ADVANCED ANALYSIS ===\n")
    
    code_example = """
from metrics_tracker import MetricsTracker
from metrics_comparison import PerformanceAnalyzer

tracker = MetricsTracker(agent_name="PPO_Agent")
tracker.load("metrics/PPO_Agent_20241019_120000.pkl")

# Detect learning plateaus
plateaus = PerformanceAnalyzer.detect_learning_plateaus(
    tracker,
    metric='total_reward',
    window_size=50,
    threshold=0.01
)
print("Learning Plateaus Detected:")
for start, end in plateaus:
    print(f"  Episodes {start} to {end}")

# Compute skill rating evolution
ratings = PerformanceAnalyzer.compute_skill_rating(
    tracker,
    initial_rating=1500.0
)
print(f"\\nFinal Skill Rating: {ratings[-1]:.1f}")

# Analyze action patterns
action_analysis = PerformanceAnalyzer.analyze_action_patterns(tracker)
print("\\nAction Effectiveness:")
for action, effectiveness in action_analysis['effectiveness_ratio'].items():
    print(f"  {action}: {effectiveness:.2f}")

# Reward decomposition
reward_decomp = PerformanceAnalyzer.compute_reward_decomposition(tracker)
print("\\nReward Sources:")
for source, percentage in reward_decomp['percentages'].items():
    print(f"  {source}: {percentage:.1f}%")

# Identify failure modes
failures = PerformanceAnalyzer.identify_failure_modes(
    tracker,
    threshold_percentile=10.0
)
print(f"\\nFailure Analysis:")
print(f"  Failure rate: {failures['failure_rate']:.2%}")
print(f"  Avg survival in failures: {failures['avg_survival_time']:.1f}")
print(f"  Self-kill rate in failures: {failures['self_kill_rate']:.2%}")
    """
    
    print(code_example)


# =============================================================================
# COMPLETE TRAINING LOOP WITH METRICS
# =============================================================================

def training_loop_with_metrics():
    """
    Complete example of a training loop with metrics tracking.
    """
    print("\n=== COMPLETE TRAINING LOOP WITH METRICS ===\n")
    
    code_example = """
from metrics_tracker import MetricsTracker
import numpy as np

# Initialize
agent = YourAgent()  # PPO, DQN, LLM, etc.
metrics_tracker = MetricsTracker(
    agent_name="MyAgent",
    save_dir="metrics"
)

NUM_EPISODES = 1000
SAVE_INTERVAL = 100

for episode in range(NUM_EPISODES):
    # Start episode
    game_state = env.reset()
    opponent_types = env.get_opponent_types()
    map_name = env.get_map_name()
    
    metrics_tracker.start_episode(
        episode_id=episode,
        opponent_types=opponent_types,
        map_name=map_name
    )
    
    done = False
    step = 0
    
    # Episode loop
    while not done:
        # Select action
        action = agent.act(game_state)
        
        # Record action
        metrics_tracker.record_action(action, is_valid=True)
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Record events
        events = info.get('events', [])
        for event in events:
            # Compute event-specific reward
            event_reward = compute_event_reward(event)
            metrics_tracker.record_event(event, reward=event_reward)
        
        # Agent learning (if applicable)
        agent.learn(game_state, action, reward, next_state, done)
        
        game_state = next_state
        step += 1
    
    # End episode
    won = info.get('won', False)
    rank = info.get('rank', 1)
    total_steps = info.get('total_steps', step)
    
    metrics_tracker.end_episode(
        won=won,
        rank=rank,
        survival_steps=step,
        total_steps=total_steps,
        metadata={
            'episode': episode,
            'epsilon': getattr(agent, 'epsilon', None)  # For epsilon-greedy agents
        }
    )
    
    # Periodic saving and reporting
    if (episode + 1) % SAVE_INTERVAL == 0:
        metrics_tracker.save()
        
        # Print summary
        summary = metrics_tracker.get_summary_stats(last_n=SAVE_INTERVAL)
        print(f"\\nEpisodes {episode+1-SAVE_INTERVAL+1}-{episode+1}:")
        print(f"  Win Rate: {summary['win_rate']:.3f}")
        print(f"  Avg Reward: {summary['avg_reward']:.2f}")
        print(f"  Avg Survival: {summary['avg_survival_time']:.1f}")

# Final save
metrics_tracker.save()
print("\\nTraining complete! Metrics saved.")
    """
    
    print(code_example)


# =============================================================================
# USAGE WITH DIFFERENT AGENT TYPES
# =============================================================================

def llm_agent_integration():
    """
    Example: Integrating with an LLM-based agent.
    """
    print("\n=== LLM AGENT INTEGRATION ===\n")
    
    code_example = """
from metrics_tracker import MetricsTracker

class LLMAgent:
    def __init__(self, model_name="gpt-4"):
        self.model_name = model_name
        self.metrics_tracker = MetricsTracker(
            agent_name=f"LLM_{model_name}",
            save_dir="metrics"
        )
        
    def act(self, game_state):
        # Get action from LLM
        prompt = self.create_prompt(game_state)
        response = self.query_llm(prompt)
        action = self.parse_action(response)
        
        # Track action
        if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
            self.metrics_tracker.record_action(
                action,
                is_valid=self.is_valid_action(action, game_state)
            )
        
        return action
    
    def track_reasoning(self, reasoning_text):
        '''Track LLM reasoning for analysis'''
        if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
            if self.metrics_tracker.current_episode.metadata is None:
                self.metrics_tracker.current_episode.metadata = {}
            
            # Store reasoning samples
            if 'reasoning_samples' not in self.metrics_tracker.current_episode.metadata:
                self.metrics_tracker.current_episode.metadata['reasoning_samples'] = []
            
            self.metrics_tracker.current_episode.metadata['reasoning_samples'].append(
                reasoning_text
            )
    """
    
    print(code_example)


def rl_agent_comparison():
    """
    Example: Comparing multiple RL algorithms.
    """
    print("\n=== RL ALGORITHM COMPARISON ===\n")
    
    code_example = """
# Train multiple agents
from metrics_tracker import MetricsTracker

algorithms = [
    ('PPO', PPOAgent()),
    ('DQN', DQNAgent()),
    ('A2C', A2CAgent()),
    ('SAC', SACAgent())
]

trackers = {}

for name, agent in algorithms:
    tracker = MetricsTracker(agent_name=name, save_dir="metrics")
    trackers[name] = tracker
    
    # Train agent
    for episode in range(NUM_EPISODES):
        tracker.start_episode(episode_id=episode)
        # ... training loop ...
        tracker.end_episode(won, rank, survival_steps, total_steps)
    
    tracker.save()

# Compare algorithms
from metrics_comparison import AgentComparator

comparator = AgentComparator(list(trackers.values()))

# Statistical comparison
pairwise_results = comparator.pairwise_comparison(
    metric='total_reward',
    test='mannwhitney'
)

# Find best algorithm
win_rates = comparator.compare_win_rates()
best_agent = max(win_rates.items(), key=lambda x: x[1]['win_rate'])
print(f"Best Algorithm: {best_agent[0]} with {best_agent[1]['win_rate']:.3f} win rate")

# Sample efficiency comparison
efficiency = comparator.compute_sample_efficiency(target_performance=0.6)
most_efficient = min(
    [(a, e) for a, e in efficiency.items() if e is not None],
    key=lambda x: x[1]
)
print(f"Most Sample Efficient: {most_efficient[0]} ({most_efficient[1]} episodes)")
    """
    
    print(code_example)


# =============================================================================
# ROBUSTNESS TESTING
# =============================================================================

def robustness_testing_example():
    """
    Example: Testing agent robustness across different conditions.
    """
    print("\n=== ROBUSTNESS TESTING ===\n")
    
    code_example = """
from metrics_tracker import MetricsTracker

agent = YourAgent()
tracker = MetricsTracker(agent_name="RobustAgent", save_dir="metrics")

# Test conditions
opponent_types = ['rule_based', 'random', 'aggressive', 'defensive']
map_names = ['classic', 'small', 'large', 'maze']
scenarios = ['coins_only', 'pvp', 'crates', 'mixed']

episode_id = 0

# Test across all conditions
for opponent in opponent_types:
    for map_name in map_names:
        for scenario in scenarios:
            for trial in range(10):  # 10 trials per condition
                episode_id += 1
                
                # Setup environment with specific conditions
                env.setup(
                    opponent_types=[opponent],
                    map_name=map_name,
                    scenario=scenario
                )
                
                # Start tracking
                tracker.start_episode(
                    episode_id=episode_id,
                    opponent_types=[opponent],
                    map_name=map_name,
                    scenario=scenario
                )
                
                # Run episode
                # ... episode loop ...
                
                tracker.end_episode(won, rank, survival_steps, total_steps)

# Analyze robustness
tracker.save()

# Performance by condition
opp_performance = tracker.compare_by_opponent()
map_performance = tracker.compare_by_map()

print("Robustness Analysis:")
print(f"  Opponent variance: {np.std([d['win_rate'] for d in opp_performance.values()]):.3f}")
print(f"  Map variance: {np.std([d['win_rate'] for d in map_performance.values()]):.3f}")

# Find weaknesses
worst_opponent = min(opp_performance.items(), key=lambda x: x[1]['win_rate'])
worst_map = min(map_performance.items(), key=lambda x: x[1]['win_rate'])

print(f"\\nWeakest vs: {worst_opponent[0]} ({worst_opponent[1]['win_rate']:.3f} WR)")
print(f"Weakest map: {worst_map[0]} ({worst_map[1]['win_rate']:.3f} WR)")
    """
    
    print(code_example)


# =============================================================================
# MAIN DEMO
# =============================================================================

def main():
    """
    Run all integration examples.
    """
    print("=" * 80)
    print("METRICS SYSTEM INTEGRATION GUIDE")
    print("=" * 80)
    
    integrate_with_callbacks()
    integrate_with_act()
    integrate_with_training()
    
    evaluate_single_agent_example()
    compare_multiple_agents_example()
    advanced_analysis_example()
    
    training_loop_with_metrics()
    
    llm_agent_integration()
    rl_agent_comparison()
    robustness_testing_example()
    
    print("\n" + "=" * 80)
    print("QUICK START CHECKLIST")
    print("=" * 80)
    print("""
1. Import MetricsTracker in your agent files:
   from metrics_tracker import MetricsTracker

2. Initialize in setup():
   self.metrics_tracker = MetricsTracker(agent_name="YourAgent")

3. Start episode at beginning of each round:
   self.metrics_tracker.start_episode(episode_id, opponent_types, map_name)

4. Track actions in act():
   self.metrics_tracker.record_action(action, is_valid=True)

5. Track events as they occur:
   self.metrics_tracker.record_event(event, reward=event_reward)

6. End episode at round end:
   self.metrics_tracker.end_episode(won, rank, survival_steps, total_steps)

7. Save periodically:
   if episode % 100 == 0:
       self.metrics_tracker.save()

8. Compare agents:
   from metrics_comparison import load_and_compare_agents
   comparator = load_and_compare_agents(
       ["metrics/agent1.pkl", "metrics/agent2.pkl"]
   )

9. Generate visualizations:
   from metrics_visualization import generate_all_plots
   generate_all_plots(trackers, comparator, save_dir="plots")

10. Analyze results:
    summary = tracker.get_summary_stats()
    print(f"Win Rate: {summary['win_rate']:.3f}")
    """)
    
    print("\n" + "=" * 80)
    print("For more details, see the documentation in each module:")
    print("  - metrics_tracker.py: Core metrics collection")
    print("  - metrics_comparison.py: Statistical analysis and comparison")
    print("  - metrics_visualization.py: Plotting and visualization")
    print("=" * 80)


if __name__ == "__main__":
    main()
