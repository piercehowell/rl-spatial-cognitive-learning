import wandb
import numpy as np
def get_average_correlation_coeff(dist_correlation_coeff_data):

    idx_ranges = [[0,39], [40,79], [80,119], [120,159],[160,199]]
    for start_idx, end_idx in idx_ranges:
        subset = dist_correlation_coeff_data[start_idx:end_idx+1]
        mean = np.mean(subset)
        std_dev = np.std(subset)
        print(f"Distance Correlation for [{start_idx}, {end_idx}]: Mean={mean}, STD={std_dev}")


project = "rl_spatial_cognitive_learning"
group="ppo_no_lstm+mlp_extractor.policy_net.1"
run_id = "decent-donkey-63"

api = wandb.Api()
run = api.run('star-lab-gt/rl_spatial_cognitive_learning/0yg64wsq')
dist_correlation_coeff_data = []
if run.state=='finished':
    print(run.history())
    for i, row in run.history().iterrows():
        dist_correlation_coeff_data.append(row["eval/dist_correlation_coeff"])

get_average_correlation_coeff(dist_correlation_coeff_data)