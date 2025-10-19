"""
Agent Comparison and Statistical Analysis Module

Provides tools for comparing multiple agents, statistical significance testing,
and generating comparison reports.
"""
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
import json
from collections import defaultdict


class AgentComparator:
    """
    Compare performance of multiple agents with statistical analysis.
    """
    
    def __init__(self, trackers: List['MetricsTracker']):
        """
        Initialize comparator with multiple agent trackers.
        
        Args:
            trackers: List of MetricsTracker objects for different agents
        """
        self.trackers = {t.agent_name: t for t in trackers}
        self.agent_names = list(self.trackers.keys())
    
    def compare_win_rates(self, confidence: float = 0.95) -> Dict[str, Any]:
        """
        Compare win rates across agents with confidence intervals.
        
        Args:
            confidence: Confidence level for intervals (default: 0.95)
            
        Returns:
            Comparison results with confidence intervals
        """
        results = {}
        
        for name, tracker in self.trackers.items():
            if not tracker.episodes:
                continue
            
            wins = [1 if ep.won else 0 for ep in tracker.episodes]
            n = len(wins)
            win_rate = np.mean(wins)
            
            # Compute confidence interval using Wilson score
            z = stats.norm.ppf((1 + confidence) / 2)
            
            if n > 0:
                # Wilson score interval
                denominator = 1 + z**2 / n
                center = (win_rate + z**2 / (2*n)) / denominator
                margin = z * np.sqrt((win_rate * (1 - win_rate) / n + z**2 / (4*n**2))) / denominator
                
                ci_lower = center - margin
                ci_upper = center + margin
            else:
                ci_lower = ci_upper = win_rate
            
            results[name] = {
                'win_rate': win_rate,
                'num_episodes': n,
                'wins': sum(wins),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'confidence_level': confidence
            }
        
        return results
    
    def pairwise_comparison(self, metric: str = 'total_reward', 
                           test: str = 'mannwhitney') -> Dict[str, Dict[str, Any]]:
        """
        Perform pairwise statistical comparisons between agents.
        
        Args:
            metric: Metric to compare ('total_reward', 'survival_time', etc.)
            test: Statistical test ('mannwhitney', 'ttest', 'bootstrap')
            
        Returns:
            Dictionary of pairwise comparison results
        """
        results = {}
        
        # Extract metric values for each agent
        agent_values = {}
        for name, tracker in self.trackers.items():
            values = []
            for ep in tracker.episodes:
                if metric == 'total_reward':
                    values.append(ep.total_reward)
                elif metric == 'survival_time':
                    values.append(ep.survival_time)
                elif metric == 'won':
                    values.append(1.0 if ep.won else 0.0)
                elif metric == 'coins_collected':
                    values.append(ep.coins_collected)
                elif metric == 'opponents_killed':
                    values.append(ep.opponents_killed)
                else:
                    values.append(ep.metadata.get(metric, 0.0))
            agent_values[name] = np.array(values)
        
        # Pairwise comparisons
        for i, agent1 in enumerate(self.agent_names):
            for agent2 in self.agent_names[i+1:]:
                pair_key = f"{agent1}_vs_{agent2}"
                
                values1 = agent_values[agent1]
                values2 = agent_values[agent2]
                
                if len(values1) == 0 or len(values2) == 0:
                    continue
                
                # Compute statistics
                mean_diff = np.mean(values1) - np.mean(values2)
                
                # Perform statistical test
                if test == 'mannwhitney':
                    statistic, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                    test_name = "Mann-Whitney U"
                elif test == 'ttest':
                    statistic, p_value = stats.ttest_ind(values1, values2)
                    test_name = "Independent t-test"
                elif test == 'bootstrap':
                    # Bootstrap confidence interval for mean difference
                    statistic, p_value, ci = self._bootstrap_comparison(values1, values2)
                    test_name = "Bootstrap"
                else:
                    raise ValueError(f"Unknown test: {test}")
                
                results[pair_key] = {
                    'agent1': agent1,
                    'agent2': agent2,
                    'agent1_mean': np.mean(values1),
                    'agent2_mean': np.mean(values2),
                    'mean_difference': mean_diff,
                    'agent1_std': np.std(values1),
                    'agent2_std': np.std(values2),
                    'test': test_name,
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'significant_at_0.05': p_value < 0.05,
                    'significant_at_0.01': p_value < 0.01,
                }
        
        return results
    
    def _bootstrap_comparison(self, values1: np.ndarray, values2: np.ndarray,
                             n_bootstrap: int = 10000, confidence: float = 0.95) -> Tuple[float, float, Tuple[float, float]]:
        """
        Bootstrap comparison of two samples.
        
        Args:
            values1: First sample
            values2: Second sample
            n_bootstrap: Number of bootstrap iterations
            confidence: Confidence level
            
        Returns:
            (statistic, p_value, confidence_interval)
        """
        observed_diff = np.mean(values1) - np.mean(values2)
        
        # Bootstrap
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            sample1 = np.random.choice(values1, size=len(values1), replace=True)
            sample2 = np.random.choice(values2, size=len(values2), replace=True)
            bootstrap_diffs.append(np.mean(sample1) - np.mean(sample2))
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Compute p-value (two-tailed)
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        # Confidence interval
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
        
        return observed_diff, p_value, (ci_lower, ci_upper)
    
    def compute_sample_efficiency(self, target_performance: float = 0.5,
                                  metric: str = 'win_rate') -> Dict[str, int]:
        """
        Compute sample efficiency: episodes needed to reach target performance.
        
        Args:
            target_performance: Target performance threshold
            metric: Metric to evaluate ('win_rate', 'avg_reward')
            
        Returns:
            Dictionary mapping agent name to episodes required
        """
        results = {}
        
        for name, tracker in self.trackers.items():
            episodes = tracker.episodes
            if not episodes:
                results[name] = None
                continue
            
            # Compute rolling performance
            window_size = 10  # Use 10-episode window
            for i in range(window_size, len(episodes) + 1):
                window = episodes[i-window_size:i]
                
                if metric == 'win_rate':
                    performance = np.mean([1 if ep.won else 0 for ep in window])
                elif metric == 'avg_reward':
                    performance = np.mean([ep.total_reward for ep in window])
                else:
                    performance = 0.0
                
                if performance >= target_performance:
                    results[name] = i  # Episodes needed
                    break
            else:
                results[name] = None  # Target not reached
        
        return results
    
    def compute_robustness_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute robustness metrics: performance variance across conditions.
        
        Returns:
            Dictionary with robustness metrics per agent
        """
        results = {}
        
        for name, tracker in self.trackers.items():
            if not tracker.episodes:
                continue
            
            # Performance variance across opponents
            opp_performance = tracker.compare_by_opponent()
            opp_win_rates = [data['win_rate'] for data in opp_performance.values() if data['num_episodes'] >= 5]
            
            # Performance variance across maps
            map_performance = tracker.compare_by_map()
            map_win_rates = [data['win_rate'] for data in map_performance.values() if data['num_episodes'] >= 5]
            
            # Overall performance stability (coefficient of variation)
            rewards = [ep.total_reward for ep in tracker.episodes]
            survival_times = [ep.survival_time for ep in tracker.episodes]
            
            results[name] = {
                'opponent_variance': np.std(opp_win_rates) if len(opp_win_rates) > 1 else 0.0,
                'map_variance': np.std(map_win_rates) if len(map_win_rates) > 1 else 0.0,
                'reward_cv': np.std(rewards) / (np.mean(rewards) + 1e-8),  # Coefficient of variation
                'survival_cv': np.std(survival_times) / (np.mean(survival_times) + 1e-8),
                'num_opponent_types': len(opp_performance),
                'num_maps': len(map_performance),
            }
        
        return results
    
    def generate_comparison_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive comparison report.
        
        Args:
            output_file: Optional file to save report
            
        Returns:
            Report as string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("AGENT COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 1. Summary statistics
        report_lines.append("1. SUMMARY STATISTICS")
        report_lines.append("-" * 80)
        
        for name, tracker in self.trackers.items():
            stats = tracker.get_summary_stats()
            report_lines.append(f"\nAgent: {name}")
            report_lines.append(f"  Episodes: {stats.get('num_episodes', 0)}")
            report_lines.append(f"  Win Rate: {stats.get('win_rate', 0):.3f} ± {stats.get('win_rate_std', 0):.3f}")
            report_lines.append(f"  Avg Reward: {stats.get('avg_reward', 0):.2f} ± {stats.get('reward_std', 0):.2f}")
            report_lines.append(f"  Avg Survival Time: {stats.get('avg_survival_time', 0):.1f} ± {stats.get('survival_time_std', 0):.1f}")
            report_lines.append(f"  Avg Kills/Episode: {stats.get('avg_kills_per_episode', 0):.2f}")
            report_lines.append(f"  Self-Kill Rate: {stats.get('self_kill_rate', 0):.3f}")
            report_lines.append(f"  Bomb Effectiveness: {stats.get('bomb_effectiveness', 0):.3f}")
            report_lines.append(f"  Invalid Action Rate: {stats.get('invalid_action_rate', 0):.3f}")
        
        report_lines.append("")
        
        # 2. Win rate comparison with confidence intervals
        report_lines.append("2. WIN RATE COMPARISON (95% Confidence Intervals)")
        report_lines.append("-" * 80)
        
        win_rate_comp = self.compare_win_rates()
        
        # Sort by win rate
        sorted_agents = sorted(win_rate_comp.items(), 
                             key=lambda x: x[1]['win_rate'], 
                             reverse=True)
        
        for rank, (name, data) in enumerate(sorted_agents, 1):
            wr = data['win_rate']
            ci_lower = data['ci_lower']
            ci_upper = data['ci_upper']
            report_lines.append(
                f"{rank}. {name:20s} {wr:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] "
                f"({data['wins']}/{data['num_episodes']} wins)"
            )
        
        report_lines.append("")
        
        # 3. Statistical significance tests
        report_lines.append("3. PAIRWISE STATISTICAL COMPARISONS (Total Reward)")
        report_lines.append("-" * 80)
        
        pairwise = self.pairwise_comparison(metric='total_reward', test='mannwhitney')
        
        for pair, results in pairwise.items():
            sig_marker = "***" if results['significant_at_0.01'] else ("**" if results['significant_at_0.05'] else "")
            report_lines.append(
                f"{results['agent1']:15s} vs {results['agent2']:15s} "
                f"| Δ={results['mean_difference']:8.2f} | p={results['p_value']:.4f} {sig_marker}"
            )
        
        report_lines.append("")
        report_lines.append("Significance: *** p<0.01, ** p<0.05")
        report_lines.append("")
        
        # 4. Sample efficiency
        report_lines.append("4. SAMPLE EFFICIENCY (Episodes to 50% Win Rate)")
        report_lines.append("-" * 80)
        
        efficiency = self.compute_sample_efficiency(target_performance=0.5, metric='win_rate')
        
        sorted_efficiency = sorted([(name, eps) for name, eps in efficiency.items() if eps is not None],
                                  key=lambda x: x[1])
        
        for name, episodes in sorted_efficiency:
            report_lines.append(f"{name:20s} {episodes:6d} episodes")
        
        # List agents that didn't reach target
        not_reached = [name for name, eps in efficiency.items() if eps is None]
        if not_reached:
            report_lines.append(f"\nDid not reach target: {', '.join(not_reached)}")
        
        report_lines.append("")
        
        # 5. Robustness metrics
        report_lines.append("5. ROBUSTNESS METRICS")
        report_lines.append("-" * 80)
        
        robustness = self.compute_robustness_metrics()
        
        for name, metrics in robustness.items():
            report_lines.append(f"\n{name}:")
            report_lines.append(f"  Opponent Variance: {metrics['opponent_variance']:.4f}")
            report_lines.append(f"  Map Variance: {metrics['map_variance']:.4f}")
            report_lines.append(f"  Reward CV: {metrics['reward_cv']:.4f}")
            report_lines.append(f"  Survival CV: {metrics['survival_cv']:.4f}")
            report_lines.append(f"  Tested on {metrics['num_opponent_types']} opponent types, {metrics['num_maps']} maps")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to {output_file}")
        
        return report
    
    def export_comparison_json(self, output_file: str):
        """
        Export comparison data as JSON for further analysis.
        
        Args:
            output_file: JSON file to write
        """
        data = {
            'agents': list(self.agent_names),
            'summary_stats': {name: tracker.get_summary_stats() 
                            for name, tracker in self.trackers.items()},
            'win_rate_comparison': self.compare_win_rates(),
            'pairwise_comparisons': self.pairwise_comparison(metric='total_reward'),
            'sample_efficiency': self.compute_sample_efficiency(),
            'robustness': self.compute_robustness_metrics(),
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Comparison data exported to {output_file}")


class PerformanceAnalyzer:
    """
    Advanced performance analysis tools.
    """
    
    @staticmethod
    def detect_learning_plateaus(tracker: 'MetricsTracker', 
                                 metric: str = 'total_reward',
                                 window_size: int = 50,
                                 threshold: float = 0.01) -> List[Tuple[int, int]]:
        """
        Detect learning plateaus where improvement stagnates.
        
        Args:
            tracker: MetricsTracker to analyze
            metric: Metric to analyze
            window_size: Window for computing slopes
            threshold: Threshold for detecting plateaus (slope magnitude)
            
        Returns:
            List of (start_episode, end_episode) tuples for plateaus
        """
        _, values, _ = tracker.get_learning_curve(metric=metric, window_size=10)
        
        if len(values) < window_size * 2:
            return []
        
        plateaus = []
        in_plateau = False
        plateau_start = 0
        
        for i in range(window_size, len(values) - window_size):
            # Compute slope over window
            window = values[i-window_size:i+window_size]
            x = np.arange(len(window))
            slope = np.polyfit(x, window, 1)[0]
            
            if abs(slope) < threshold:
                if not in_plateau:
                    in_plateau = True
                    plateau_start = i
            else:
                if in_plateau:
                    plateaus.append((plateau_start, i))
                    in_plateau = False
        
        # Handle ongoing plateau
        if in_plateau:
            plateaus.append((plateau_start, len(values)))
        
        return plateaus
    
    @staticmethod
    def compute_skill_rating(tracker: 'MetricsTracker', 
                            initial_rating: float = 1500.0,
                            k_factor: float = 32.0) -> List[float]:
        """
        Compute Elo-style skill rating evolution over episodes.
        
        Args:
            tracker: MetricsTracker to analyze
            initial_rating: Starting Elo rating
            k_factor: K-factor for rating updates
            
        Returns:
            List of ratings after each episode
        """
        ratings = [initial_rating]
        current_rating = initial_rating
        
        for ep in tracker.episodes:
            # Expected score (simplified: assume opponent has same rating)
            expected = 0.5
            
            # Actual score
            if ep.won:
                actual = 1.0
            elif ep.rank == len(ep.opponent_types) + 1:  # Last place
                actual = 0.0
            else:
                # Interpolate based on rank
                actual = 1.0 - (ep.rank - 1) / len(ep.opponent_types)
            
            # Update rating
            current_rating = current_rating + k_factor * (actual - expected)
            ratings.append(current_rating)
        
        return ratings
    
    @staticmethod
    def analyze_action_patterns(tracker: 'MetricsTracker') -> Dict[str, Any]:
        """
        Analyze action selection patterns.
        
        Args:
            tracker: MetricsTracker to analyze
            
        Returns:
            Dictionary with action pattern statistics
        """
        # Aggregate action distributions
        total_actions = defaultdict(int)
        win_actions = defaultdict(int)
        loss_actions = defaultdict(int)
        
        for ep in tracker.episodes:
            for action, count in ep.action_distribution.items():
                total_actions[action] += count
                if ep.won:
                    win_actions[action] += count
                else:
                    loss_actions[action] += count
        
        # Compute frequencies
        total_count = sum(total_actions.values())
        action_freq = {action: count / total_count for action, count in total_actions.items()}
        
        # Compute win-specific frequencies
        win_total = sum(win_actions.values())
        loss_total = sum(loss_actions.values())
        
        win_freq = {action: count / win_total for action, count in win_actions.items()} if win_total > 0 else {}
        loss_freq = {action: count / loss_total for action, count in loss_actions.items()} if loss_total > 0 else {}
        
        # Compute action effectiveness (win_freq / loss_freq ratio)
        effectiveness = {}
        for action in total_actions.keys():
            wf = win_freq.get(action, 0)
            lf = loss_freq.get(action, 0)
            if lf > 0:
                effectiveness[action] = wf / lf
            else:
                effectiveness[action] = float('inf') if wf > 0 else 1.0
        
        return {
            'overall_frequency': action_freq,
            'win_frequency': win_freq,
            'loss_frequency': loss_freq,
            'effectiveness_ratio': effectiveness,
            'total_actions': sum(total_actions.values()),
        }
    
    @staticmethod
    def compute_reward_decomposition(tracker: 'MetricsTracker') -> Dict[str, float]:
        """
        Decompose total reward into contributions from different sources.
        
        Args:
            tracker: MetricsTracker to analyze
            
        Returns:
            Dictionary mapping reward sources to their contribution percentages
        """
        total_rewards = defaultdict(float)
        overall_total = 0.0
        
        for ep in tracker.episodes:
            for source, reward in ep.reward_breakdown.items():
                total_rewards[source] += reward
                overall_total += reward
        
        # Compute percentages
        if overall_total != 0:
            percentages = {source: (reward / overall_total) * 100 
                          for source, reward in total_rewards.items()}
        else:
            percentages = {}
        
        return {
            'percentages': percentages,
            'absolute_values': dict(total_rewards),
            'total_reward': overall_total,
        }
    
    @staticmethod
    def identify_failure_modes(tracker: 'MetricsTracker', 
                               threshold_percentile: float = 10.0) -> Dict[str, Any]:
        """
        Identify common failure modes (worst-performing episodes).
        
        Args:
            tracker: MetricsTracker to analyze
            threshold_percentile: Percentile threshold for "failure"
            
        Returns:
            Analysis of failure modes
        """
        # Get reward threshold for failures
        rewards = [ep.total_reward for ep in tracker.episodes]
        threshold = np.percentile(rewards, threshold_percentile)
        
        # Identify failure episodes
        failures = [ep for ep in tracker.episodes if ep.total_reward <= threshold]
        
        if not failures:
            return {'num_failures': 0}
        
        # Analyze common characteristics
        avg_survival = np.mean([ep.survival_time for ep in failures])
        self_kill_rate = np.mean([1 if ep.self_kills > 0 else 0 for ep in failures])
        avg_invalid = np.mean([ep.invalid_actions for ep in failures])
        
        # Most common opponent types in failures
        opponent_counts = defaultdict(int)
        for ep in failures:
            for opp in ep.opponent_types:
                opponent_counts[opp] += 1
        
        return {
            'num_failures': len(failures),
            'threshold_reward': threshold,
            'avg_survival_time': avg_survival,
            'self_kill_rate': self_kill_rate,
            'avg_invalid_actions': avg_invalid,
            'common_opponents': dict(opponent_counts),
            'failure_rate': len(failures) / len(tracker.episodes),
        }


def load_and_compare_agents(metrics_files: List[str], 
                            output_dir: str = "comparison_results") -> AgentComparator:
    """
    Load multiple agent metrics and create comparator.
    
    Args:
        metrics_files: List of pickle files containing MetricsTracker data
        output_dir: Directory to save comparison results
        
    Returns:
        AgentComparator object
    """
    from metrics_tracker import MetricsTracker
    
    os.makedirs(output_dir, exist_ok=True)
    
    trackers = []
    for filepath in metrics_files:
        tracker = MetricsTracker(agent_name="temp")
        tracker.load(filepath)
        trackers.append(tracker)
    
    comparator = AgentComparator(trackers)
    
    # Generate report
    report = comparator.generate_comparison_report(
        output_file=os.path.join(output_dir, "comparison_report.txt")
    )
    print(report)
    
    # Export JSON
    comparator.export_comparison_json(
        output_file=os.path.join(output_dir, "comparison_data.json")
    )
    
    return comparator