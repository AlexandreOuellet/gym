import glob
import json

from multiprocessing import Process

def main():
    experiments = {}
    for x in glob.glob('./experiments.json'):
        json_data=open(x).read()
        data = json.loads(json_data)
        experiments = data["experiments"]

    env_params = experiments['env']
    agent_params = experiments['agent']

    variations_by_type = {}
    for variation in experiments['variations']:
        for key, value in variation.items():
            if key not in variations_by_type:
                variations_by_type[key] = []
            variations_by_type[key].append(value)

    all_variation_type = list(variations_by_type)

    baseline_experiment = {**env_params, **agent_params}
    all_experiments = []
    all_experiments.append(baseline_experiment)
    if len(all_variation_type) != 0:
        all_experiments += buildExperiments([], variations_by_type, all_variation_type.copy(), baseline_experiment)

    processes = []
    while len(all_experiments) != 0:
        while len(all_experiments) > 0 and len(processes) < 4:
            experiment = all_experiments.pop()
            p = Process(target=runExperiment, args=(experiment,))
            p.start()
            processes.append(p)
        
        while len(processes) != 0:
            p = processes.pop()
            p.join()

    
    # for experiment in all_experiments:
    #     p = Process(target=runExperiment, args=(experiment,))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()


def runExperiment(experiment):
    import numpy as np
    from collections import deque
    import gym
    from gym.wrappers import Monitor
    from agents.dqnagent import DQNAgent

    #environment parameters
    gym_id = experiment["gym_id"]
    sliding_window_solved_score = experiment["sliding_window_solved_score"]
    sliding_window_score_length = experiment["sliding_window_score_length"]
    env_seed = experiment["env_seed"]
    max_episode = experiment["max_episode"]

    env = gym.make(gym_id)
    env = Monitor(env, "{}".format(experiment['folder']), video_callable=False, force=True, resume=False,
            write_upon_reset=False, uid=None, mode=None)

    env.seed(env_seed)
    scores = deque()
    sw_scores = deque(maxlen=sliding_window_score_length)

    #agent parameters
    agent_seed = experiment["agent_seed"]
    activation = experiment["activation"]
    min_episode_before_acting = experiment["min_episode_before_acting"]
    epsilon = experiment["epsilon"]
    nb_hidden_layer = experiment["nb_hidden_layer"]
    layer_width = experiment["layer_width"]
    memory_length = experiment["memory_length"]
    batch_size = experiment["batch_size"]
    agent = DQNAgent(env.observation_space, env.action_space, agent_seed, min_episode_before_acting, activation, epsilon, layer_width, nb_hidden_layer, memory_length)

    current_episode = 0
    while (len(sw_scores) == 0 or np.mean(sw_scores) < sliding_window_solved_score) and (max_episode == None or current_episode < max_episode):
        state = env.reset()

        current_episode += 1
        reward = 0
        done = False
        episode_score = 0

        while not done:
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            episode_score += reward

            # if np.mean(sw_scores) > 180:
            #     env.render()

            if done:
                scores.append(episode_score)
                sw_scores.append(episode_score)

                print('Episode: {}\t Epsilon: {}\t Score: {}\t Mean Score:{}\t Sliding Score:{}\t'.format(current_episode, agent.epsilon, episode_score, np.mean(scores), np.mean(sw_scores)))
                agent.train(batch_size=batch_size)
    env.close()


def buildExperiments(all_experiments:[], variations:[], variation_types:list, current_experiment:{}):
    variation_type = variation_types.pop()
    current_experiment1 = current_experiment.copy()
    current_experiment1["folder"] += "/{}/".format(variation_type)

    for value in variations[variation_type]:
        current_experiment1 = {**current_experiment1, **{variation_type:value}}
        current_experiment2 = current_experiment1.copy()
        current_experiment2["folder"] += "/{}/".format(value)

        if len(variation_types) == 0:
            all_experiments.append(current_experiment2)
        else:
            buildExperiments(all_experiments, variations, variation_types.copy(), current_experiment2)

    return all_experiments
    
if __name__ == '__main__':
    main()