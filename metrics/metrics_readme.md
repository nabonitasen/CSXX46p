# Bomberman Agent Metrics System

A comprehensive, agent-agnostic metrics tracking and evaluation system for comparing RL and LLM agents in the Bomberman environment.

## Features

### 📊 Comprehensive Metrics Collection
- **Success Metrics**: Win rate, ranking, survival time
- **Combat Metrics**: Kills, deaths, self-kills, bomb effectiveness
- **Resource Metrics**: Coins collected, crates destroyed
- **Action Metrics**: Action distribution, invalid action rate
- **Reward Metrics**: Total reward, reward decomposition by source

### 📈 Advanced Analysis Tools
- Statistical significance testing (Mann-Whitney U, t-tests, bootstrap)
- Learning curve analysis with plateau detection
- Sample efficiency comparison
- Robustness testing across opponents/maps
- Elo-style skill rating evolution
- Action pattern analysis
- Failure mode identification

### 🎨 Rich Visualizations
- Learning curves with confidence intervals
- Multi-agent comparison charts
- Radar charts for multi-metric comparison
- Distribution plots (violin, box)
- Heatmaps for action patterns
- Performance breakdowns by opponent/map
- Comprehensive dashboards

### 🔧 Agent-Agnostic Design
Works with any agent type:
- RL algorithms (PPO, DQN, A2C, SAC, etc.)
- LLM-based agents
- Rule-based agents
- Hybrid approaches

## Installation

```bash
# Required dependencies
pip install numpy scipy matplotlib seaborn

# Optional for advanced features
pip install pandas torch
```

## Quick Start

### 1. Basic Integration

```python
from metrics_tracker import MetricsTracker

# Initialize tracker
tracker = MetricsTracker(agent_name="MyAgent", save_dir="metrics")

# Start episode
tracker.start_episode(
    episode_id=0,
    opponent_types=["rule_based"],
    map_name="classic"
)

# During episode: track actions
tracker.record_action("UP", is_valid=True)
tracker.record_action("BOMB", is_valid=True)

# During episode: track events
tracker.record_event("COIN_COLLECTED", reward=10)
tracker.record_event("KILLED_OPPONENT", reward=50)

# End episode
tracker.end_episode(
    won=True,
    rank=1,
    survival_steps=250,
    total_steps=400
)

# Save metrics
tracker.save()
```

### 2. Integration with Existing Agent

```python
# In your callbacks.py setup()
from metrics_tracker import MetricsTracker

def setup(self):
    self.train_agent = PPOAgent(...)
    self.metrics_tracker = MetricsTracker(
        agent_name="PPO_Agent",
        save_dir="metrics"
    )
    self.episode_id = 0

# In your act() function
def act(self, game_state):
    action = self.train_agent.select_action(...)
    
    if hasattr(self, 'metrics_tracker'):
        self.metrics_tracker.record_action(action, is_valid=True)
    
    return action

# In your train.py game_events_occurred()
def game_events_occurred(self, old_game_state, self_action, 
                         new_game_state, events):
    for event in events:
        reward = self.compute_event_reward(event)
        self.metrics_tracker.record_event(event, reward=reward)

# In your train.py end_of_round()
def end_of_round(self, last_game_state, last_action, events):
    self.metrics_tracker.end_episode(
        won="WON" in events,
        rank=self.get_rank(),
        survival_steps=self.current_step,
        total_steps=400
    )
    
    if self.round_counter % 100 == 0:
        self.metrics_tracker.save()
```

### 3. Compare Multiple Agents

```python
from metrics_tracker import MetricsTracker
from metrics_comparison import AgentComparator
from metrics_visualization import generate_all_plots

# Load multiple agents
trackers = []
for agent_file in ["ppo.pkl", "dqn.pkl", "llm.pkl"]:
    tracker = MetricsTracker(agent_name="temp")
    tracker.load(f"metrics/{agent_file}")
    trackers.append(tracker)

# Create comparator
comparator = AgentComparator(trackers)

# Generate comprehensive report
report = comparator.generate_comparison_report(
    output_file="comparison_results/report.txt"
)
print(report)

# Statistical comparison
pairwise = comparator.pairwise_comparison(metric='total_reward')
for pair, results in pairwise.items():
    if results['significant_at_0.05']:
        print(f"{pair}: p={results['p_value']:.4f} ***")

# Generate all visualizations
generate_all_plots(trackers, comparator, save_dir="plots")
```

## API Reference

### MetricsTracker

Main class for tracking metrics during agent execution.

#### Methods

**`__init__(agent_name, save_dir="metrics")`**
- Initialize tracker for a specific agent

**`start_episode(episode_id, opponent_types, map_name, scenario)`**
- Begin tracking a new episode

**`record_action(action, is_valid=True)`**
- Record an action taken by the agent

**`record_event(event, reward=0.0)`**
- Record a game event with associated reward

**`end_episode(won, rank, survival_steps, total_steps, metadata=None)`**
- Finalize episode and compute derived metrics

**`get_summary_stats(last_n=None)`**
- Get statistical summary over all or recent episodes

**`get_learning_curve(metric, window_size=10)`**
- Get learning curve data for plotting

**`compare_by_opponent()`**
- Compare performance against different opponent types

**`compare_by_map()`**
- Compare performance across different maps

**`save(filename=None)`**
- Save metrics to disk (pickle format)

**`load(filename)`**
- Load metrics from disk

**`export_to_csv(filename=None)`**
- Export episode data as CSV

### AgentComparator

Compare multiple agents with statistical analysis.

#### Methods

**`__init__(trackers)`**
- Initialize with list of MetricsTracker objects

**`compare_win_rates(confidence=0.95)`**
- Compare win rates with confidence intervals

**`pairwise_comparison(metric, test='mannwhitney')`**
- Perform pairwise statistical tests between agents

**`compute_sample_efficiency(target_performance, metric)`**
- Calculate episodes needed to reach target performance

**`compute_robustness_metrics()`**
- Analyze performance variance across conditions

**`generate_comparison_report(output_file=None)`**
- Generate comprehensive text report

**`export_comparison_json(output_file)`**
- Export comparison data as JSON

### PerformanceAnalyzer

Advanced performance analysis tools.

#### Static Methods

**`detect_learning_plateaus(tracker, metric, window_size, threshold)`**
- Identify episodes where learning stagnates

**`compute_skill_rating(tracker, initial_rating=1500.0)`**
- Calculate Elo-style skill rating evolution

**`analyze_action_patterns(tracker)`**
- Analyze action selection patterns and effectiveness

**`compute_reward_decomposition(tracker)`**
- Break down reward sources by percentage

**`identify_failure_modes(tracker, threshold_percentile=10.0)`**
- Analyze characteristics of worst-performing episodes

### MetricsVisualizer

Create visualizations for metrics.

#### Methods

**`plot_learning_curves(trackers, metric, window_size, save_name)`**
- Plot learning curves for multiple agents

**`plot_win_rate_comparison(comparator, save_name)`**
- Bar chart of win rates with confidence intervals

**`plot_metric_distributions(trackers, metric, save_name)`**
- Violin plots showing metric distributions

**`plot_radar_chart(trackers, metrics, save_name)`**
- Radar chart for multi-metric comparison

**`plot_sample_efficiency(comparator, target_performance, save_name)`**
- Horizontal bar chart of sample efficiency

**`plot_robustness_comparison(comparator, save_name)`**
- Compare robustness metrics across agents

**`plot_performance_vs_opponents(tracker, save_name)`**
- Performance breakdown by opponent type

**`plot_action_heatmap(tracker, save_name)`**
- Heatmap of action frequencies in wins vs losses

**`plot_reward_decomposition(tracker, save_name)`**
- Pie chart of reward sources

**`plot_skill_rating_evolution(trackers, save_name)`**
- Line plot of Elo-style ratings over time

**`create_comprehensive_dashboard(trackers, comparator, save_name)`**
- Multi-panel dashboard with key metrics

## Metrics Collected

### Episode-Level Metrics

| Metric | Description |
|--------|-------------|
| `won` | Boolean indicating episode win |
| `rank` | Final ranking (1 = first place) |
| `survival_time` | Steps survived before elimination |
| `total_steps` | Total episode length |
| `opponents_killed` | Number of opponents eliminated |
| `self_kills` | Number of self-eliminations |
| `bombs_placed` | Total bombs placed |
| `bombs_hit_opponent` | Bombs that damaged/killed opponent |
| `bombs_destroyed_crate` | Bombs that destroyed crates |
| `coins_collected` | Coins collected |
| `crates_destroyed` | Crates destroyed |
| `total_actions` | Total actions taken |
| `invalid_actions` | Number of invalid actions |
| `action_distribution` | Frequency of each action type |
| `total_reward` | Cumulative reward |
| `reward_breakdown` | Reward by source |

### Aggregate Statistics

| Statistic | Description |
|-----------|-------------|
| `win_rate` | Percentage of episodes won |
| `avg_survival_time` | Mean survival time |
| `avg_reward` | Mean total reward per episode |
| `avg_kills_per_episode` | Mean opponents killed |
| `self_kill_rate` | Self-kills per episode |
| `bomb_effectiveness` | Ratio of effective bombs to total |
| `invalid_action_rate` | Percentage of invalid actions |
| `coins_per_step` | Resource collection efficiency |

## Example Outputs

### Summary Statistics
```
Agent: PPO_Agent
  Episodes: 1000
  Win Rate: 0.652 ± 0.048
  Avg Reward: 245.32 ± 89.17
  Avg Survival Time: 312.5 ± 67.8
  Avg Kills/Episode: 1.23
  Self-Kill Rate: 0.087
  Bomb Effectiveness: 0.342
  Invalid Action Rate: 0.023
```

### Comparison Report
```
=================================================================================
AGENT COMPARISON REPORT
=================================================================================

1. SUMMARY STATISTICS
---------------------------------------------------------------------------------

Agent: PPO_Agent
  Episodes: 1000
  Win Rate: 0.652 ± 0.048
  ...

2. WIN RATE COMPARISON (95% Confidence Intervals)
---------------------------------------------------------------------------------
1. PPO_Agent             0.652 [0.621, 0.682] (652/1000 wins)
2. LLM_Agent             0.534 [0.502, 0.566] (534/1000 wins)
3. DQN_Agent             0.487 [0.455, 0.519] (487/1000 wins)

3. PAIRWISE STATISTICAL COMPARISONS (Total Reward)
---------------------------------------------------------------------------------
PPO_Agent       vs DQN_Agent        | Δ=  42.15 | p=0.0023 **
PPO_Agent       vs LLM_Agent        | Δ=  28.73 | p=0.0451 *
DQN_Agent       vs LLM_Agent        | Δ= -13.42 | p=0.2341

Significance: *** p<0.01, ** p<0.05

4. SAMPLE EFFICIENCY (Episodes to 50% Win Rate)
---------------------------------------------------------------------------------
PPO_Agent             143 episodes
DQN_Agent             287 episodes
LLM_Agent             412 episodes
```

## Best Practices

### 1. Consistent Event Tracking
Always track the same events across agents for fair comparison:

```python
# Define standard event rewards
EVENT_REWARDS = {
    'COIN_COLLECTED': 10,
    'KILLED_OPPONENT': 50,
    'KILLED_SELF': -100,
    'CRATE_DESTROYED': 5,
    # ... etc
}

for event in events:
    reward = EVENT_REWARDS.get(event, 0)
    tracker.record_event(event, reward=reward)
```

### 2. Regular Checkpointing
Save metrics periodically to avoid data loss:

```python
if episode % 100 == 0:
    tracker.save(f"checkpoint_ep{episode}.pkl")
```

### 3. Metadata Usage
Store additional context for later analysis:

```python
tracker.end_episode(
    won=won,
    rank=rank,
    survival_steps=steps,
    total_steps=total,
    metadata={
        'learning_rate': agent.lr,
        'epsilon': agent.epsilon,
        'model_version': 'v2.1',
        'notes': 'After hyperparameter tuning'
    }
)
```

### 4. Multiple Test Conditions
Test robustness across varied conditions:

```python
for opponent in ['random', 'aggressive', 'defensive']:
    for map_name in ['small', 'medium', 'large']:
        for trial in range(10):
            tracker.start_episode(
                episode_id=episode_id,
                opponent_types=[opponent],
                map_name=map_name
            )
            # ... run episode ...
```

## Troubleshooting

### Issue: Missing Episodes
**Problem**: Some episodes not tracked
**Solution**: Ensure `start_episode()` and `end_episode()` are always paired

### Issue: Dimension Mismatch in Comparisons
**Problem**: Agents have different numbers of episodes
**Solution**: Use `last_n` parameter to compare recent performance

```python
summary = tracker.get_summary_stats(last_n=100)
```

### Issue: Memory Usage
**Problem**: Too many episodes stored in memory
**Solution**: Save and clear periodically

```python
if len(tracker.episodes) > 1000:
    tracker.save()
    tracker.episodes = tracker.episodes[-100:]  # Keep recent
```

## Citation

If you use this metrics system in your research, please cite:

```bibtex
@software{bomberman_metrics_2024,
  title={Bomberman Agent Metrics System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourrepo}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Support

For issues, questions, or feature requests:
- GitHub Issues: https://github.com/yourrepo/issues
- Email: your.email@example.com

## Changelog

### Version 1.0.0 (2024-10-19)
- Initial release
- Core metrics tracking
- Statistical comparison tools
- Visualization suite
- Comprehensive documentation
