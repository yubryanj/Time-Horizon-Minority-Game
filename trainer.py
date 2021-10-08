from agents import Optimizing_Agent, Random_Agent
from copy import deepcopy
from env import Minority_Game
import numpy as np
from tqdm import tqdm
from utils import get_configs

def main():

    # Retrieve the configs
    configs = get_configs()
    
    # Instantiate the environment
    environment = Minority_Game(
        configs = configs,
    )

    # Initialize the agents
    agents = [Optimizing_Agent(configs) for i in range(configs.get('n_agents'))]

    scores = np.zeros(configs.get('n_agents'))

    # Reset the environment
    observations = environment.reset()

    for timestep in tqdm(range(configs.get('max_steps'))):

        # Compute the joint actions from all agents
        joint_action = [agent.compute_action(observations) for agent in agents]

        # Store the past_observation for computing valuations
        previous_observation = deepcopy(observations)

        # take a step(a.k.a transition) in the environment
        observations, rewards, done, info = environment.step(joint_action)    

        # Keep track of agents' scores
        scores += rewards

        # Each agent updates their individual valuation
        for agent in agents:
            agent.update_valuation(
                previous_observation, 
                info
            )
         
    print(scores)



if __name__ == "__main__":
    main()