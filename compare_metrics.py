from metrics.metrics_tracker import MetricsTracker
from metrics.metrics_comparison import AgentComparator
from metrics.metrics_visualization import generate_all_plots

import pickle
data = pickle.load(open("agent_code/q_learning/metrics/q_learning_3.pkl", "rb"))
data = pickle.load(open("agent_code/ppo/metrics/ppo_3.pkl", "rb"))
# data.get("agent_name")
# data["agent_name"] = "ppo"
# data.get('episodes')[0]['agent_name'] = "ppo"
# data.get('episodes')[1]['agent_name'] = "ppo"
# data.get('episodes')[2]['agent_name'] = "ppo"

# with open("agent_code/ppo/metrics/ppo_3.pkl", "wb") as f:
#     pickle.dump(data, f)


# Load multiple agents
trackers = []
metrics_dictionary = {
    'q_learning': 'Q Learning_8000.pkl',
    # 'ppo': 'ppo_3.pkl',
}
for agent, file in metrics_dictionary.items():
    tracker = MetricsTracker(agent_name=agent, save_dir = f"agent_code/{agent}/metrics")
    tracker.load(file)
    trackers.append(tracker)


# Create comparator
comparator = AgentComparator(trackers)

# Generate comprehensive report
report = comparator.generate_comparison_report(
    output_file="metrics/comparison_results/q_learning.txt"
)
print(report)

# Statistical comparison
pairwise = comparator.pairwise_comparison(metric='total_reward')
for pair, results in pairwise.items():
    if results['significant_at_0.05']:
        print(f"{pair}: p={results['p_value']:.4f} ***")

# Generate all visualizations
generate_all_plots(trackers, comparator, save_dir="plots")