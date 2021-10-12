from agents import Optimizing_Agent, Random_Agent
from copy import deepcopy
from env import Minority_Game
import numpy as np
from tqdm import tqdm
from utils import get_configs, Logger, process_data

def main():

    # Retrieve the configs
    configs, experiment_id = get_configs()

    # Initialize the logger
    logger = Logger(configs)
    
    # Instantiate the environment
    environment = Minority_Game(
        configs = configs,
    )

    for sample in tqdm(range(configs.get('n_samples'))):
        
        # Initialize the agents
        if configs.get('agent_type') == 'optimizing':
            agents = [Optimizing_Agent(configs) for i in range(configs.get('n_agents'))]
        elif configs.get('agent_type') == 'random':
            agents = [Random_Agent(configs) for i in range(configs.get('n_agents'))]

        # Total aggregated_score
        actual_scores = np.zeros(configs.get('n_agents'))

        # THMG_scores should be set to virtual scores - no need to keep separate score set
        # THMG Score - starts counting after THMG horizon
        THMG_scores = np.zeros(configs.get('n_agents'))

        # Reset the environment
        observations = environment.reset()
      
        """ Main Training Loop """
        for timestep in range(configs.get('max_steps')):

            # Compute the joint actions from all agents
            actions = [agent.compute_action(observations) for agent in agents]

            # Store the past_observation for computing valuations
            previous_observation = deepcopy(observations)

            # take a step(a.k.a transition) in the environment
            observations, rewards, done, info = environment.step(actions)    

            # Log the statistics before updating values
            logger.log_before_updates({
                'actions': actions,
                'agents': agents,
                'observations': previous_observation,
                'samples': sample,
                'rewards': rewards,
            })

            # Keep track of agents' scores
            actual_scores += rewards

            # Update the THMG score after passing the horizon line
            if timestep > configs.get('THMG_horizon'):
                THMG_scores += rewards

            # Each agent updates their individual valuation
            for agent in agents:
                agent.update_valuation(
                    previous_observation, 
                    info
                )    

            # Log the statistics after updating values
            logger.log_after_update({
                'agents': agents,
                'samples': sample,
                'actual_scores': actual_scores,
                'THMG_scores': THMG_scores

            })
         
    # Output the logs to a csv file
    logger.dump(experiment_id)

    # Process the results
    process_data(experiment_id)


if __name__ == "__main__":
    main()