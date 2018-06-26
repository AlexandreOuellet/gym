import glob
import json

import gym
from gym.wrappers import Monitor

import uuid

# from agents.dqnagent import DQNAgent

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

    all_variation_type = variations_by_type.keys()

    all_experiments = buildExperiments([], all_variation_type, variations_by_type, variations_by_type.copy(), {})



        # experiment = {**experiment, **variation}
        # experiment = {**experiment, **{"id":"{}".format(uuid.uuid4())}}

        # variations.append(experiment)

    for experiment in all_experiments:
        env = gym.make(experiment['gym'])
        env = Monitor(env, "./monitoring/{}/".format(experiment['id']), video_callable=False, force=True, resume=False,
                write_upon_reset=False, uid=None, mode=None)
        

def buildExperiments(all_experiments:[], variations:[], variation_types:[]):
    current_experiment = {}
    for variation_type in variation_types:
        for value in variations[variation_type]:
            current_experiment = {**current_experiment, **{variation_type:value}}
        all_experiments.append(current_experiment.copy())
        
    # for variation in all_variations:
    #     experiment = {}
    #     experiment = {**experiment, **experiments['env']}
    #     experiment = {**experiment, **experiments['agent']}
    return all_experiments
    
if __name__ == '__main__':
    main()