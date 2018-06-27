import glob
import json

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

stats_files = []

for x in glob.glob('./monitoring/**/*.stats.json', recursive=True):
    stats_files.append(x)

episode_rewards = []


for stats_file in stats_files:
    json_data=open(stats_file).read()
    data = json.loads(json_data)
    episode_rewards.append(data['episode_rewards'])

for stats in episode_rewards:
    plt.plot(stats)
plt.show()