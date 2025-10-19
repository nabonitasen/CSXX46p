"""
Visualization Tools for Agent Metrics

Provides plotting functions for comparing agents and visualizing performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional, Tuple
import os


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class MetricsVisualizer:
    """
    Create visualizations for agent performance metrics.
    """
    
    def __init__(self, save_dir: str = "plots"):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    @staticmethod
    def _safe_error_bars(means: List[float], ci_lowers: List[float], 
                        ci_uppers: List[float]) -> Tuple[List[float], List[float]]:
        """
        Safely compute error bars, ensuring they're always non-negative.
        
        Args:
            means: Mean values
            ci_lowers: Lower confidence bounds
            ci_uppers: Upper confidence bounds
            
        Returns:
            (errors_lower, errors_upper) tuple with non-negative values
        """
        errors_lower = []
        errors_upper = []
        
        for i in range(len(means)):
            # Ensure non-negative error bars
            err_lower = max(0.0, means[i] - ci_lowers[i])
            err_upper = max(0.0, ci_uppers[i] - means[i])
            
            # If error is extremely small (< 1e-10), treat as zero
            err_lower = 0.0 if err_lower < 1e-10 else err_lower
            err_upper = 0.0 if err_upper < 1e-10 else err_upper
            
            errors_lower.append(err_lower)
            errors_upper.append(err_upper)
        
        return errors_lower, errors_upper
    
    def plot_learning_curves(self, trackers: List['MetricsTracker'],
                            metric: str = 'total_reward',
                            window_size: int = 10,
                            save_name: Optional[str] = None):
        """
        Plot learning curves for multiple agents.
        
        Args:
            trackers: List of MetricsTracker objects
            metric: Metric to plot
            window_size: Moving average window
            save_name: Filename to save plot
        """
        plt.figure(figsize=(12, 6))
        
        for tracker in trackers:
            episodes, values, moving_avg = tracker.get_learning_curve(
                metric=metric, window_size=window_size
            )
            
            if not episodes:
                continue
            
            # Plot raw values with transparency
            plt.plot(episodes, values, alpha=0.2, linewidth=0.5)
            
            # Plot moving average
            plt.plot(episodes, moving_avg, label=tracker.agent_name, linewidth=2)
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.title(f'Learning Curves: {metric.replace("_", " ").title()}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_win_rate_comparison(self, comparator: 'AgentComparator',
                                save_name: Optional[str] = None):
        """
        Plot win rate comparison with confidence intervals.
        
        Args:
            comparator: AgentComparator object
            save_name: Filename to save plot
        """
        win_rates = comparator.compare_win_rates()
        
        if not win_rates:
            print("No win rate data available for plotting")
            return
        
        agents = list(win_rates.keys())
        means = [win_rates[agent]['win_rate'] for agent in agents]
        ci_lowers = [win_rates[agent]['ci_lower'] for agent in agents]
        ci_uppers = [win_rates[agent]['ci_upper'] for agent in agents]
        
        # Safely compute error bars
        errors_lower, errors_upper = self._safe_error_bars(means, ci_lowers, ci_uppers)
        
        # Sort by win rate
        sorted_indices = np.argsort(means)[::-1]
        agents = [agents[i] for i in sorted_indices]
        means = [means[i] for i in sorted_indices]
        errors_lower = [errors_lower[i] for i in sorted_indices]
        errors_upper = [errors_upper[i] for i in sorted_indices]
        
        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(agents))
        
        # Check if we have meaningful error bars
        has_errors = any(e > 0 for e in errors_lower) or any(e > 0 for e in errors_upper)
        
        if has_errors:
            plt.bar(x_pos, means, yerr=[errors_lower, errors_upper],
                   capsize=5, alpha=0.7, edgecolor='black', error_kw={'linewidth': 2})
        else:
            # No error bars if all are zero (happens with very few samples)
            plt.bar(x_pos, means, alpha=0.7, edgecolor='black')
            plt.text(0.5, 0.95, 'Note: No confidence intervals (insufficient data)',
                    transform=plt.gca().transAxes, ha='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        plt.xlabel('Agent', fontsize=12)
        plt.ylabel('Win Rate', fontsize=12)
        plt.title('Win Rate Comparison (95% CI)', fontsize=14)
        plt.xticks(x_pos, agents, rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_metric_distributions(self, trackers: List['MetricsTracker'],
                                 metric: str = 'total_reward',
                                 save_name: Optional[str] = None):
        """
        Plot distribution of a metric across agents using violin plots.
        
        Args:
            trackers: List of MetricsTracker objects
            metric: Metric to plot
            save_name: Filename to save plot
        """
        plt.figure(figsize=(12, 6))
        
        data = []
        labels = []
        
        for tracker in trackers:
            values = []
            for ep in tracker.episodes:
                if metric == 'total_reward':
                    values.append(ep.total_reward)
                elif metric == 'survival_time':
                    values.append(ep.survival_time)
                elif metric == 'coins_collected':
                    values.append(ep.coins_collected)
                elif metric == 'opponents_killed':
                    values.append(ep.opponents_killed)
                else:
                    values.append(ep.metadata.get(metric, 0.0))
            
            if values:
                data.append(values)
                labels.append(tracker.agent_name)
        
        if data:
            plt.violinplot(data, showmeans=True, showmedians=True)
            plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
            plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
            plt.title(f'Distribution: {metric.replace("_", " ").title()}', fontsize=14)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def plot_radar_chart(self, trackers: List['MetricsTracker'],
                        metrics: List[str] = None,
                        save_name: Optional[str] = None):
        """
        Create radar chart comparing agents across multiple metrics.
        
        Args:
            trackers: List of MetricsTracker objects
            metrics: List of metrics to include
            save_name: Filename to save plot
        """
        if metrics is None:
            metrics = ['win_rate', 'avg_survival_time', 'avg_kills_per_episode',
                      'bomb_effectiveness', 'avg_coins_per_episode']
        
        # Get summary stats for each agent
        agent_stats = {}
        for tracker in trackers:
            stats = tracker.get_summary_stats()
            agent_stats[tracker.agent_name] = stats
        
        # Normalize metrics to 0-1 scale
        normalized_data = {}
        max_values = {metric: 0.0 for metric in metrics}
        
        # Find max values for normalization
        for agent_name, stats in agent_stats.items():
            for metric in metrics:
                value = stats.get(metric, 0.0)
                max_values[metric] = max(max_values[metric], value)
        
        # Normalize
        for agent_name, stats in agent_stats.items():
            normalized_data[agent_name] = []
            for metric in metrics:
                value = stats.get(metric, 0.0)
                max_val = max_values[metric]
                normalized = value / max_val if max_val > 0 else 0.0
                normalized_data[agent_name].append(normalized)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for agent_name, values in normalized_data.items():
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=agent_name)
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.title('Multi-Metric Agent Comparison', fontsize=14, pad=20)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_sample_efficiency(self, comparator: 'AgentComparator',
                              target_performance: float = 0.5,
                              save_name: Optional[str] = None):
        """
        Plot sample efficiency comparison.
        
        Args:
            comparator: AgentComparator object
            target_performance: Target performance threshold
            save_name: Filename to save plot
        """
        efficiency = comparator.compute_sample_efficiency(
            target_performance=target_performance,
            metric='win_rate'
        )
        
        # Filter out agents that didn't reach target
        reached = {agent: eps for agent, eps in efficiency.items() if eps is not None}
        
        if not reached:
            print("No agents reached target performance")
            return
        
        agents = list(reached.keys())
        episodes = list(reached.values())
        
        # Sort by efficiency (fewer episodes = better)
        sorted_indices = np.argsort(episodes)
        agents = [agents[i] for i in sorted_indices]
        episodes = [episodes[i] for i in sorted_indices]
        
        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(agents))
        colors = plt.cm.viridis(np.linspace(0, 1, len(agents)))
        
        plt.barh(x_pos, episodes, color=colors, alpha=0.7, edgecolor='black')
        
        plt.yticks(x_pos, agents)
        plt.xlabel('Episodes Required', fontsize=12)
        plt.ylabel('Agent', fontsize=12)
        plt.title(f'Sample Efficiency (Episodes to {target_performance*100:.0f}% Win Rate)', fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_action_heatmap(self, tracker: 'MetricsTracker',
                           save_name: Optional[str] = None):
        """
        Plot heatmap of action frequencies in wins vs losses.
        
        Args:
            tracker: MetricsTracker object
            save_name: Filename to save plot
        """
        from metrics.metrics_comparison import PerformanceAnalyzer
        
        analysis = PerformanceAnalyzer.analyze_action_patterns(tracker)
        
        actions = list(analysis['overall_frequency'].keys())
        win_freq = [analysis['win_frequency'].get(a, 0) for a in actions]
        loss_freq = [analysis['loss_frequency'].get(a, 0) for a in actions]
        
        data = np.array([win_freq, loss_freq])
        
        plt.figure(figsize=(10, 4))
        sns.heatmap(data, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=actions, yticklabels=['Wins', 'Losses'],
                   cbar_kws={'label': 'Frequency'})
        
        plt.title(f'Action Distribution: {tracker.agent_name}', fontsize=14)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_robustness_comparison(self, comparator: 'AgentComparator',
                                  save_name: Optional[str] = None):
        """
        Plot robustness metrics comparison.
        
        Args:
            comparator: AgentComparator object
            save_name: Filename to save plot
        """
        robustness = comparator.compute_robustness_metrics()
        
        agents = list(robustness.keys())
        opp_var = [robustness[a]['opponent_variance'] for a in agents]
        map_var = [robustness[a]['map_variance'] for a in agents]
        reward_cv = [robustness[a]['reward_cv'] for a in agents]
        
        x = np.arange(len(agents))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width, opp_var, width, label='Opponent Variance', alpha=0.8)
        ax.bar(x, map_var, width, label='Map Variance', alpha=0.8)
        ax.bar(x + width, reward_cv, width, label='Reward CV', alpha=0.8)
        
        ax.set_xlabel('Agent', fontsize=12)
        ax.set_ylabel('Variance / CV', fontsize=12)
        ax.set_title('Robustness Metrics (Lower = More Robust)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_performance_vs_opponents(self, tracker: 'MetricsTracker',
                                     save_name: Optional[str] = None):
        """
        Plot performance breakdown by opponent type.
        
        Args:
            tracker: MetricsTracker object
            save_name: Filename to save plot
        """
        opp_comparison = tracker.compare_by_opponent()
        
        if not opp_comparison:
            print("No opponent data available")
            return
        
        opponents = list(opp_comparison.keys())
        win_rates = [opp_comparison[o]['win_rate'] for o in opponents]
        num_episodes = [opp_comparison[o]['num_episodes'] for o in opponents]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Win rates
        colors = plt.cm.viridis(np.linspace(0, 1, len(opponents)))
        ax1.bar(range(len(opponents)), win_rates, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Opponent Type', fontsize=12)
        ax1.set_ylabel('Win Rate', fontsize=12)
        ax1.set_title(f'{tracker.agent_name}: Win Rate vs Opponents', fontsize=13)
        ax1.set_xticks(range(len(opponents)))
        ax1.set_xticklabels(opponents, rotation=45, ha='right')
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Episode counts
        ax2.bar(range(len(opponents)), num_episodes, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Opponent Type', fontsize=12)
        ax2.set_ylabel('Number of Episodes', fontsize=12)
        ax2.set_title('Episodes per Opponent Type', fontsize=13)
        ax2.set_xticks(range(len(opponents)))
        ax2.set_xticklabels(opponents, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_reward_decomposition(self, tracker: 'MetricsTracker',
                                 save_name: Optional[str] = None):
        """
        Plot pie chart of reward sources.
        
        Args:
            tracker: MetricsTracker object
            save_name: Filename to save plot
        """
        from metrics.metrics_comparison import PerformanceAnalyzer
        
        decomposition = PerformanceAnalyzer.compute_reward_decomposition(tracker)
        percentages = decomposition['percentages']
        
        if not percentages:
            print("No reward data available")
            return
        
        # Filter out very small contributions
        filtered = {k: v for k, v in percentages.items() if abs(v) > 0.5}
        other = sum(v for k, v in percentages.items() if abs(v) <= 0.5)
        if other != 0:
            filtered['Other'] = other
        
        labels = list(filtered.keys())
        sizes = list(filtered.values())
        
        # Separate positive and negative rewards
        positive_mask = np.array(sizes) > 0
        colors = ['green' if p else 'red' for p in positive_mask]
        
        # Make colors lighter (simulate alpha effect)
        import matplotlib.colors as mcolors
        light_colors = []
        for color in colors:
            rgb = mcolors.to_rgb(color)
            # Blend with white to simulate alpha=0.7
            light_rgb = tuple(0.3 + 0.7 * c for c in rgb)
            light_colors.append(light_rgb)
        
        plt.figure(figsize=(10, 8))
        
        # Create pie chart (without alpha parameter for compatibility)
        wedges, texts, autotexts = plt.pie(
            np.abs(sizes), 
            labels=labels, 
            autopct='%1.1f%%',
            colors=light_colors, 
            startangle=90
        )
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.title(f'Reward Decomposition: {tracker.agent_name}', fontsize=14)
        
        # Add legend for positive/negative
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', label='Positive'),
            Patch(facecolor='lightcoral', label='Negative')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_skill_rating_evolution(self, trackers: List['MetricsTracker'],
                                   save_name: Optional[str] = None):
        """
        Plot Elo-style skill rating evolution over time.
        
        Args:
            trackers: List of MetricsTracker objects
            save_name: Filename to save plot
        """
        from metrics.metrics_comparison import PerformanceAnalyzer
        
        plt.figure(figsize=(12, 6))
        
        for tracker in trackers:
            if not tracker.episodes:
                continue
            
            ratings = PerformanceAnalyzer.compute_skill_rating(tracker)
            episodes = list(range(len(ratings)))
            
            plt.plot(episodes, ratings, label=tracker.agent_name, linewidth=2, alpha=0.8)
        
        plt.axhline(y=1500, color='gray', linestyle='--', alpha=0.5, label='Initial Rating')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Skill Rating (Elo-style)', fontsize=12)
        plt.title('Skill Rating Evolution', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_comprehensive_dashboard(self, trackers: List['MetricsTracker'],
                                      comparator: 'AgentComparator',
                                      save_name: Optional[str] = None):
        """
        Create a comprehensive dashboard with multiple subplots.
        
        Args:
            trackers: List of MetricsTracker objects
            comparator: AgentComparator object
            save_name: Filename to save plot
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Learning curves (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        for tracker in trackers:
            episodes, _, moving_avg = tracker.get_learning_curve(metric='total_reward', window_size=10)
            if episodes:
                ax1.plot(episodes, moving_avg, label=tracker.agent_name, linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Learning Curves')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Win rates (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        win_rates = comparator.compare_win_rates()
        agents = list(win_rates.keys())
        means = [win_rates[a]['win_rate'] for a in agents]
        ax2.bar(range(len(agents)), means, alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(agents)))
        ax2.set_xticklabels(agents, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Win Rate')
        ax2.set_title('Win Rate Comparison')
        ax2.set_ylim(0, 1.0)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Sample efficiency (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        efficiency = comparator.compute_sample_efficiency(target_performance=0.5)
        reached = {a: e for a, e in efficiency.items() if e is not None}
        if reached:
            agents_eff = list(reached.keys())
            episodes_eff = list(reached.values())
            ax3.barh(range(len(agents_eff)), episodes_eff, alpha=0.7, edgecolor='black')
            ax3.set_yticks(range(len(agents_eff)))
            ax3.set_yticklabels(agents_eff, fontsize=8)
            ax3.set_xlabel('Episodes')
            ax3.set_title('Sample Efficiency (50% WR)')
            ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Survival time distribution (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        for tracker in trackers:
            survival_times = [ep.survival_time for ep in tracker.episodes]
            if survival_times:
                ax4.hist(survival_times, alpha=0.5, label=tracker.agent_name, bins=20)
        ax4.set_xlabel('Survival Time')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Survival Time Distribution')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Combat metrics (middle middle)
        ax5 = fig.add_subplot(gs[1, 1])
        agents_combat = []
        kills = []
        self_kills = []
        for tracker in trackers:
            stats = tracker.get_summary_stats()
            agents_combat.append(tracker.agent_name)
            kills.append(stats.get('avg_kills_per_episode', 0))
            self_kills.append(stats.get('self_kill_rate', 0))
        
        x = np.arange(len(agents_combat))
        width = 0.35
        ax5.bar(x - width/2, kills, width, label='Avg Kills', alpha=0.7)
        ax5.bar(x + width/2, self_kills, width, label='Self-Kill Rate', alpha=0.7)
        ax5.set_xticks(x)
        ax5.set_xticklabels(agents_combat, rotation=45, ha='right', fontsize=8)
        ax5.set_ylabel('Value')
        ax5.set_title('Combat Metrics')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Robustness (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        robustness = comparator.compute_robustness_metrics()
        agents_rob = list(robustness.keys())
        reward_cv = [robustness[a]['reward_cv'] for a in agents_rob]
        ax6.bar(range(len(agents_rob)), reward_cv, alpha=0.7, edgecolor='black')
        ax6.set_xticks(range(len(agents_rob)))
        ax6.set_xticklabels(agents_rob, rotation=45, ha='right', fontsize=8)
        ax6.set_ylabel('Coefficient of Variation')
        ax6.set_title('Reward Stability (Lower = Better)')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. Bomb effectiveness (bottom left)
        ax7 = fig.add_subplot(gs[2, 0])
        agents_bomb = []
        bomb_eff = []
        for tracker in trackers:
            stats = tracker.get_summary_stats()
            agents_bomb.append(tracker.agent_name)
            bomb_eff.append(stats.get('bomb_effectiveness', 0))
        ax7.bar(range(len(agents_bomb)), bomb_eff, alpha=0.7, edgecolor='black')
        ax7.set_xticks(range(len(agents_bomb)))
        ax7.set_xticklabels(agents_bomb, rotation=45, ha='right', fontsize=8)
        ax7.set_ylabel('Effectiveness')
        ax7.set_title('Bomb Effectiveness')
        ax7.set_ylim(0, 1.0)
        ax7.grid(True, alpha=0.3, axis='y')
        
        # 8. Resource collection (bottom middle)
        ax8 = fig.add_subplot(gs[2, 1])
        agents_res = []
        coins = []
        crates = []
        for tracker in trackers:
            stats = tracker.get_summary_stats()
            agents_res.append(tracker.agent_name)
            coins.append(stats.get('avg_coins_per_episode', 0))
            crates.append(stats.get('avg_crates_per_episode', 0))
        
        x = np.arange(len(agents_res))
        width = 0.35
        ax8.bar(x - width/2, coins, width, label='Coins', alpha=0.7)
        ax8.bar(x + width/2, crates, width, label='Crates', alpha=0.7)
        ax8.set_xticks(x)
        ax8.set_xticklabels(agents_res, rotation=45, ha='right', fontsize=8)
        ax8.set_ylabel('Average per Episode')
        ax8.set_title('Resource Collection')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3, axis='y')
        
        # 9. Summary statistics table (bottom right)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        table_data = []
        for tracker in trackers:
            stats = tracker.get_summary_stats()
            table_data.append([
                tracker.agent_name[:15],
                f"{stats.get('win_rate', 0):.2f}",
                f"{stats.get('avg_reward', 0):.1f}",
                f"{stats.get('avg_survival_time', 0):.0f}"
            ])
        
        table = ax9.table(cellText=table_data,
                         colLabels=['Agent', 'Win Rate', 'Avg Reward', 'Survival'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        ax9.set_title('Summary Statistics', fontsize=10, pad=20)
        
        plt.suptitle('Agent Performance Dashboard', fontsize=16, y=0.98)
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        
        plt.show()


def generate_all_plots(trackers: List['MetricsTracker'],
                      comparator: 'AgentComparator',
                      save_dir: str = "plots"):
    """
    Generate all standard plots for agent comparison.
    
    Args:
        trackers: List of MetricsTracker objects
        comparator: AgentComparator object
        save_dir: Directory to save plots
    """
    visualizer = MetricsVisualizer(save_dir=save_dir)
    
    print("Generating plots...")
    
    # 1. Learning curves
    print("  - Learning curves (reward)...")
    visualizer.plot_learning_curves(trackers, metric='total_reward',
                                    save_name='learning_curves_reward.png')
    
    print("  - Learning curves (win rate)...")
    visualizer.plot_learning_curves(trackers, metric='won',
                                    save_name='learning_curves_winrate.png')
    
    # 2. Win rate comparison
    print("  - Win rate comparison...")
    visualizer.plot_win_rate_comparison(comparator,
                                       save_name='win_rate_comparison.png')
    
    # 3. Metric distributions
    print("  - Reward distribution...")
    visualizer.plot_metric_distributions(trackers, metric='total_reward',
                                        save_name='reward_distribution.png')
    
    # 4. Radar chart
    print("  - Radar chart...")
    visualizer.plot_radar_chart(trackers,
                               save_name='radar_chart.png')
    
    # 5. Sample efficiency
    print("  - Sample efficiency...")
    visualizer.plot_sample_efficiency(comparator,
                                     save_name='sample_efficiency.png')
    
    # 6. Robustness
    print("  - Robustness metrics...")
    visualizer.plot_robustness_comparison(comparator,
                                         save_name='robustness_comparison.png')
    
    # 7. Skill rating evolution
    print("  - Skill rating evolution...")
    visualizer.plot_skill_rating_evolution(trackers,
                                          save_name='skill_rating.png')
    
    # 8. Comprehensive dashboard
    print("  - Comprehensive dashboard...")
    visualizer.create_comprehensive_dashboard(trackers, comparator,
                                             save_name='dashboard.png')
    
    # 9. Individual agent plots
    for tracker in trackers:
        print(f"  - Individual plots for {tracker.agent_name}...")
        
        visualizer.plot_performance_vs_opponents(tracker,
            save_name=f'{tracker.agent_name}_vs_opponents.png')
        
        visualizer.plot_action_heatmap(tracker,
            save_name=f'{tracker.agent_name}_action_heatmap.png')
        
        visualizer.plot_reward_decomposition(tracker,
            save_name=f'{tracker.agent_name}_reward_decomposition.png')
    
    print(f"\nAll plots saved to {save_dir}/")