import os
import json
import numpy as np

path = './experiments/evaluate/libero/LIBERO_10'

results = {}

# for root, dirs, files in os.walk(path):
#     for file in files:
#         if file.endswith('.json'):
#             # load json file
#             with open(os.path.join(root, file), 'r') as f:
#                 data = json.load(f)
#                 model = root.split('/')[-5]
#                 seed = root.split('/')[-2]
#                 sr = data['rollout']['overall_success_rate']
#                 results[model] = results.get(model, {})
#                 results[model][seed] = sr
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.json'):
            # load json file
            with open(os.path.join(root, file), 'r') as f:
                data = json.load(f)
                model = root.split('/')[-5]
                seed = root.split('/')[-2]
                sr_envs_dict = data['rollout_success_rate']
                sr_list = list(sr_envs_dict.values())
                results[model] = results.get(model, {})
                results[model][seed] = sr_list

print(results)

for model in results:
    model_results = []
    for seed in results[model]:
        model_results.append(results[model][seed])
    max_per_seed = np.max(model_results, axis=0)
    # breakpoint()
    mean = np.mean(max_per_seed)
    std = np.std(max_per_seed)
    print(f'{model}: {mean:.2f} ± {std:.2f}')
    