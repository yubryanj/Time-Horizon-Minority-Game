from agents import Optimizing_Agent, Random_Agent
from copy import deepcopy
from env import Minority_Game
import numpy as np
from tqdm import tqdm
from utils import get_configs, Logger, process_data

def main():

    # Retrieve the configs
    configs = get_configs()

    # Initialize the logger
    logger = Logger(configs)
    
    # Instantiate the environment
    environment = Minority_Game(
        configs = configs,
    )

    # Initialize the agents
    agents = [Optimizing_Agent(configs) for i in range(configs.get('n_agents'))]
    # agents = [Random_Agent(configs) for i in range(configs.get('n_agents'))]


    # Total aggregaed_score
    total_scores = np.zeros(configs.get('n_agents'))

    # THMG Score - starts counting after THMG horizon
    THMG_scores = np.zeros(configs.get('n_agents'))

    # Reset the environment
    observations = environment.reset()

    for timestep in tqdm(range(configs.get('max_steps'))):

        # Compute the joint actions from all agents
        actions = [agent.compute_action(observations) for agent in agents]

        # Store the past_observation for computing valuations
        previous_observation = deepcopy(observations)

        # take a step(a.k.a transition) in the environment
        observations, rewards, done, info = environment.step(actions)    

        # Log the statistics before updating values
        logger.log(
            actions         = actions,
            agents          = agents, 
            observations    = previous_observation,
            rewards         = rewards,
            total_scores    = total_scores,
            THMG_scores     = THMG_scores
        )

        # Keep track of agents' scores
        total_scores += rewards

        # Update the THMG score after passing the horizon line
        if timestep > configs.get('THMG_horizon'):
            THMG_scores += rewards

        # Each agent updates their individual valuation
        for agent in agents:
            agent.update_valuation(
                previous_observation, 
                info
            )        
         
    print(total_scores, '\n', THMG_scores)

    # Output the logs to a csv file
    logger.dump()

    # Process the results
    process_data()


if __name__ == "__main__":
    main()