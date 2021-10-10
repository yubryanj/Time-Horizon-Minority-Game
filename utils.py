import argparse
from copy import deepcopy
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()

def get_configs(
    path = './configs.json',
    ):
    """
    Retrieves the environment configuration
    :args   path    path to the config file
    :output data    configs in json format
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,   
        default='default'
    )
    args = parser.parse_args()

    with open(path) as f:
        data = f.read()
    data = json.loads(data)

    configs = data[args.experiment]

    return configs



class Logger():

    def __init__(
        self,
        configs
        ):
        self.configs = configs

        self.actions                = []
        self.hypothetical_scores    = []
        self.observations           = []
        self.rewards                = []
        self.THMG_scores            = []
        self.total_scores           = []
        self.valuations             = []
        

    def log(
        self,
        actions,
        agents, 
        observations,
        rewards,
        total_scores,
        THMG_scores,
        ):
        """
        Stores the current statistics
        """
        self.actions.append(np.array(actions))
        self.observations.append(observations)
        self.rewards.append(rewards)
        self.THMG_scores.append(deepcopy(list(THMG_scores)))
        self.total_scores.append(deepcopy(list(total_scores)))

        # Process the valuation of each strategy
        valuations = []
        hypothetical_valuation = []

        for agent in agents:
            if hasattr(agent,'valuation'):
                valuations.append(list(agent.valuation))
                hypothetical_valuation.append(list(agent.hypothetical_valuation))

            else:
                valuations.append([0 for _ in range(self.configs.get('n_strategies'))])
                hypothetical_valuation.append([0 for _ in range(self.configs.get('n_strategies'))])

        self.valuations.append(valuations)
        self.hypothetical_scores.append(hypothetical_valuation)

        pass


    def dump(
        self,
        save_dir = './logs/default.csv'
        ):
        """
        Writes the log to the filesystem
        """

        data = {
            'actions':self.actions,
            'observations':self.observations,
            'rewards':self.rewards,
            'THMG_scores':self.THMG_scores,
            'total_scores':self.total_scores,
            'valuations': self.valuations,
        }

        df = pd.DataFrame(data=data)
        df.to_csv(
            f'{save_dir}', 
            index=False
        )  


def process_data(
    log_dir = './logs/default.csv',
    save_dir = './logs/results.jpg'
    ):
    df = pd.read_csv(log_dir)

    # NOTE: Results are plotted - averaged over all agents and all strategies in one realization

    # Process the THMG scores for plotting
    data = np.array([eval(THMG_score) for THMG_score in df['THMG_scores'].to_list()])
    THMG_scores = np.mean(data,axis=1)

    # Process the valuations for plotting
    data = np.array([eval(valuation) for valuation in df['valuations'].to_list()])
    valuations = np.mean(data,axis=(1,2))

    # Plot scores
    sns.lineplot(
        data=THMG_scores, 
        legend=False
    )

    # Plot hypothetical scores
    sns.lineplot(
        data=valuations, 
        legend=False
    )

    # Save the plot
    plt.savefig(
        f'{save_dir}',
    )





if __name__=="__main__":
    process_data()