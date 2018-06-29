import glob
import json

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from collections import deque

stats_files = []

for x in glob.glob('./monitoring/**/*.stats.json', recursive=True):
    stats_files.append(x)

episode_rewards = []


for stats_file in stats_files:
    json_data=open(stats_file).read()
    data = json.loads(json_data)
    episode_rewards.append((data['episode_rewards'], stats_file))

mean_stats = []

for stats, stats_file in episode_rewards:
    all_datapoints = deque(maxlen=100)
    mean_stat_datapoint = []
    for datapoint in stats:
        all_datapoints.append(datapoint)
        mean_stat_datapoint.append(np.mean(all_datapoints))
    mean_stats.append((mean_stat_datapoint, stats, stats_file))

legends = []
# for stats, stats_file in episode_rewards:
#     mean = np.mean(stats)
#     if  mean > 65:
#         plt.plot(stats, label=mean)
#         legends.append(mean)
#     else:
#         print("Mean to low : {}".format(mean))

for mean_stats, stats, stats_file in mean_stats:
    mean = mean_stats[len(mean_stats) - 1]
    print("Mean too low.  Mean:{}\tEpisodes:{}\tFile:{}".format(mean, len(mean_stats), stats_file))
    if  mean > 194 and len(mean_stats) < 3000:
        plt.plot(mean_stats, label="{}".format(stats_file))
        legends.append(stats_file)
    # else:
    #     print("Mean too low.  Mean:{}\tEpisodes:{}".format(mean, len(mean_stats)))

plt.legend(legends)
plt.tight_layout()
plt.show()