from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):
    def __init__(
            self, 
            configs
        ):
        self.configs = configs
        super().__init__()
    
    @abstractmethod
    def compute_action(
        self,
        observation
        ):
        pass

    def update_valuation(
        self,
        observations,
        info
        ):
        pass


class Random_Agent(Agent):
    """
    The random agent selects an action from the valid actions at every timestep
    """

    def compute_action(
        self,
        observation
        ):
        """
        :args   observation observation presented to the agent in current timestep
        :output action      the agent's action for the current observation   
        """
        return np.random.choice(self.configs['actions'])


class Optimizing_Agent(Agent):

    def __init__(
        self,
        configs
        ):

        # Store the configs
        self.configs = configs

        # Endow each agent with S strategies, each of size 2^m
        self.strategies = np.random.choice(
            [0, 1], 
            size=(self.configs.get('n_strategies'), np.power(2, self.configs.get('memory')))
        )

        # Store the rolling valuations
        self.time_horizon = np.zeros((self.configs.get('tau'), self.configs.get('n_strategies')))

        # Initialize the value of each strategy to 0
        self.valuation = np.sum(self.time_horizon,axis=0)

        # Hypothetical Scores
        self.hypothetical_scores = np.zeros(self.configs.get('n_strategies'))

        pass


    def compute_action(
        self,
        observation
        ):
        
        # Identify strategies with the highest valuation. 
        best_strategies = np.where(self.valuation == np.max(self.valuation))[0]

        # Determine if a tie exists
        tie_exists = best_strategies.shape[0] > 1

        # Ties are broken via a fair coin-flip
        if tie_exists:
            strategy = np.random.choice(best_strategies)
        else:
            strategy = best_strategies[0]

        # Determine the strategy's prediction based on the history
        history = int("".join(str(x) for x in observation), 2)
        prediction = self.strategies[strategy][history]

        # The agent takes the opposite of the prediction made by the best strategy
        if prediction == 1:
            action = -1
        else:
            action = 1

        return action


    def update_valuation(
        self,
        observation,
        info
        ):

        # Convert the history to a integer for lookup
        history = int("".join(str(x) for x in observation), 2)
        
        # If a strategy predicted the outcome correctly; it gets +1; -1 otherwise
        counterfactual_scores = np.zeros(self.configs.get('n_strategies'))

        for strategy_id, strategy in enumerate(self.strategies):
            if strategy[history] == info.get('D'):
                counterfactual_scores[strategy_id] += 1
            else:
                counterfactual_scores[strategy_id] -= 1

        # Update the hypothetical score of each strategy
        self.hypothetical_scores += counterfactual_scores

        # after a "reasonable" number of iterations, begin the rolling window of cumulative strategy scores
        if info.get('timestep') > self.configs.get('THMG_horizon'):

            # Append to the time horizon buffer
            self.time_horizon[info.get('timestep') % self.configs.get('tau')] = counterfactual_scores

            # Update the valuation
            self.valuation = np.sum(self.time_horizon, axis=0)


        pass

