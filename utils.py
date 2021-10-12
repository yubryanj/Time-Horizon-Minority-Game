import argparse
from agents import Optimizing_Agent, Random_Agent
from copy import deepcopy
import json
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas.core.indexing import is_label_like
import seaborn as sns

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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

    return configs, args.experiment


class Logger():

    def __init__(
        self,
        configs
        ):
        self.configs = configs

        self.actions                = []
        self.counterfactual_rewards = []
        self.hypothetical_scores    = []
        self.observations           = []
        self.rewards                = []
        self.sample_id              = []
        self.THMG_scores            = []
        self.actual_scores          = []
        self.valuations             = []
        

    def log_before_updates(
        self,
        data
        ):
        """
        Stores the current statistics
        """
        self.actions.append(np.array(data.get('actions')))
        self.observations.append(data.get('observations'))
        self.rewards.append(list(data.get('rewards')))
        self.sample_id.append(data.get('samples'))

        pass


    def log_after_update(
        self,
        data
        ):
        
        self.THMG_scores.append(deepcopy(list(data.get('THMG_scores'))))
        self.actual_scores.append(deepcopy(list(data.get('actual_scores'))))


        agent_type = self.configs.get('agent_type')
        if agent_type == 'random':
            zeros = [0 for i in range(self.configs.get('n_agents'))]
            self.hypothetical_scores.append([zeros])
            self.valuations.append(zeros)
            self.counterfactual_rewards.append(zeros)

        elif agent_type == 'optimizing':
            hypothetical_score = []
            valuation = []
            counterfactual_rewards = []
            for strategy_id in range(self.configs.get('n_strategies')):
                score = []
                val = []
                counterfactual = []
                for agent_id in range(self.configs.get('n_agents')):
                    if isinstance(data.get('agents')[agent_id], Optimizing_Agent):
                        score.append(data.get('agents')[agent_id].hypothetical_scores[strategy_id])
                        val.append(data.get('agents')[agent_id].valuation[strategy_id])
                        counterfactual.append(data.get('agents')[agent_id].counterfactual_rewards[strategy_id])

                hypothetical_score.append(score)
                valuation.append(val)
                counterfactual_rewards.append(counterfactual)

            self.hypothetical_scores.append(hypothetical_score)
            self.valuations.append(valuation)
            self.counterfactual_rewards.append(counterfactual_rewards)

        pass


    def dump(
        self,
        experiment_id,
        ):
        """
        Writes the log to the filesystem
        """
        
        save_dir = f'./results/{experiment_id}/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data = {
            'actions':self.actions,
            'counterfactual_rewards': self.counterfactual_rewards,
            'hypothetical_scores': self.hypothetical_scores,
            'observations':self.observations,
            'rewards':self.rewards,
            'sample_id': self.sample_id,
            'THMG_scores':self.THMG_scores,
            'actual_scores':self.actual_scores,
            'valuations': self.valuations,
        }

        df = pd.DataFrame(data=data)
        df.to_csv(
            f'{save_dir}/data.csv', 
            index=False
        )  




def plot_scores(
    experiment,
    ):

    save_dir = f'./results/{experiment}'
    df = pd.read_csv(f'{save_dir}/scores.csv')

    # Plot the scores
    sns.lineplot(
        data=df, 
        x='timestep', 
        y='scores', 
        hue='label',
        alpha = 0.5
    )

    plt.title("Scores")
    plt.savefig(
        f'{save_dir}/scores.jpg',
        bbox_inches = 'tight',
        pad_inches = 0
    )
    plt.clf()  
    plt.close()  


def prepare_scores(
    experiment_id
    ):

    save_dir = f'./results/{experiment_id}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load dataset
    df = pd.read_csv(f'{save_dir}/data.csv')

    # NOTE: Results are plotted - averaged over all agents and all strategies in one sample

    # Prepare storage
    label = []
    scores = []

    # Split the data by sample(one sample is one experiment)
    samples = df['sample_id'].unique()

    for sample in samples:

        # Subset the data
        data_subset = df[df['sample_id'] == sample]

        # Process the Total Scores averaged over agents
        data = np.array([eval(score) for score in data_subset['actual_scores'].to_list()])
        actual_score = np.mean(data,axis=1)
        label.append('actual score')
        scores.append(list(actual_score))

        # Process the hypothetical scores averaged over agents and strategies
        data = np.array([eval(score) for score in data_subset['hypothetical_scores']])
        hypothetical_score = np.mean(data,axis=(1,2))
        label.append('hypothetical score')
        scores.append(list(hypothetical_score))

        pass

    n_records, n_timesteps = len(scores),len(scores[0])
    
    master_data = {
        'label':label,
        'scores':scores
    }

    # Save the processed scores
    plot_data = pd.DataFrame(data=master_data)

    # Convert each list of scores into one record for plotting
    plot_data = plot_data.explode('scores')
    # Insert the timesteps
    plot_data.insert(0,'timestep',[i for i in range(n_timesteps)]*n_records)
    plot_data.to_csv(
        f'{save_dir}/scores.csv', 
        index=False,
    )  


def compute_per_agent_change_in_wealth(
    experiment_id
    ):

    save_dir = f'./results/{experiment_id}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load dataset
    df = pd.read_csv(f'{save_dir}/data.csv')

    # NOTE: Results are plotted - averaged over all agents and all strategies in one sample

    # Prepare storage
    rewards = []

    # Split the data by sample(one sample is one experiment)
    samples = df['sample'].unique()

    for sample in samples:

        # Subset the data
        data_subset = df[df['sample'] == sample]

        # Process the THMG scores for plotting
        data = np.array([eval(record) for record in data_subset['rewards'].to_list()])
        reward = np.mean(data)

        rewards.append(reward)

    # Plot scores
    plot(
        data = rewards, 
        title = 'Mean Per Agent Per Step - Change in Wealth',
        save_dir = save_dir
    )

    print(f'Mean per step change in wealth averaged over agents: {np.mean(rewards)}')


def compute_per_strategy_change_in_wealth(
    experiment_id
    ):

    save_dir = f'./results/{experiment_id}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load dataset
    df = pd.read_csv(f'{save_dir}/data.csv')

    # Prepare storage
    counterfactual_rewards = []

    # Split the data by sample(one sample is one experiment)
    samples = df['sample'].unique()

    for sample in samples:

        # Subset the data
        data_subset = df[df['sample'] == sample]

        # Process the THMG scores for plotting
        data = np.array([eval(record) for record in data_subset['counterfactual_rewards'].to_list()])
        counterfactual_reward = np.mean(data,axis=(1,2))

        counterfactual_rewards.append(counterfactual_reward)

    # Plot scores
    plot(
        data = counterfactual_rewards, 
        title = 'Mean Per Strategy Per Step - Change in Wealth',
        save_dir = save_dir
    )

    print(f'Mean per step change in wealth averaged over strategies: {np.mean(counterfactual_rewards)}')


def process_data(
    experiment_id
    ):

    print("Preparing Scores")
    prepare_scores(experiment_id)
    print("Scores prepared.")

    print("Plotting scores")
    plot_scores(experiment_id)
    print("Plotting finished.")

    # compute_per_agent_change_in_wealth(experiment_id)

    # compute_per_strategy_change_in_wealth(experiment_id)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,   
        default='default'
    )
    args = parser.parse_args()

    process_data(args.experiment)