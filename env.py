import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from gym.spaces import Box


class Minority_Game(MultiAgentEnv):

    def __init__(
        self, 
        configs
        ):

        # Store the environment's configs
        self.configs = configs

        # History of aggregated actions, A = sum_{i=1}^{N}a_{i}(t)
        self.history = []

        # Agents take actions in [-1,1]
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(1,)
        )

        # All agents receive an observation containing the past m joint actions
        self.observation_space = gym.spaces.Dict(
            {
                "history": Box(
                    0,
                    1, 
                    shape=(self.configs.get('memory'),)
                ),
            }
        )

        pass



    def reset(
        self
        ):
        """
        Resets the environment
        """

        # Reset the timestep
        self.timestep = 0

        # Reset the history to a vector of 0's
        self.history = [0 for i in range(self.configs.get('memory'))]

        # Retrieve the observations of the resetted environment
        # All agents receive the same observation
        observations = self.history[-self.configs.get('memory'):]
        
        return observations


    def step(
        self, 
        actions
        ):
        """
        Takes one transition step in the environment
        :args actions           dictionary containing the actions decided by each agent
        :output observations    dictionary containing the next observations for each agent
        :output rewards         dictionary containing the rewards for each agent
        :output done            dictionary containing __all__ reflecting if the episode is finished
        :output info            dictionary containing any additional episode information
        """

        # Increment the timestep counter
        self.timestep += 1

        # Convert the actions into binary value, D
        # D(t) = \frac{1}{2}[Sgn(2A-N)+1] \in {0,1}
        A = np.sum(actions)
        N = self.configs.get('n_agents')
        D = 0.5 * ( np.sign(2*A-N) + 1 )

        # Store the aggregated action into the history
        self.history.append(int(D))

        # Retrieve the last m elements as the universal observation
        observations = self.history[-self.configs.get('memory'):]

        # Compute the true reward to each agent
        rewards = -np.sign(A) * np.array(actions)

        # The episode is finished after the timesteps reaches the maximum number of steps
        if self.configs.get('max_steps') != 'None':
            done = {"__all__" : self.timestep == self.configs.get('max_steps')}
        else:
            done = {"__all__" : False}

        # Return extra info
        info    = {
            'D': D,
            'timestep': self.timestep
            }

        return observations, rewards, done, info

