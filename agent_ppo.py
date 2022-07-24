""" Proximal Policy Optimization (PPO): https://openai.com/blog/openai-baselines-ppo/
    train PPO-Agent
"""


# ------------------------------ Import Libraries -----------------------------
import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

import Snake

# ------------------------------ Configurations -------------------------------
parser = argparse.ArgumentParser()

# training parameters
parser.add_argument('--gamma', type=float, default=0.99, help="discount factor")
parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help="beta factor")
parser.add_argument('--n_latent_var', type=int, default=8, help="number of variables in hidden layer")
parser.add_argument('--update_timestep', type=int, default=33, help="update policy every n timesteps")
parser.add_argument('--K_epochs', type=int, default=80, help="update policy for K epochs")
parser.add_argument('--eps_clip', type=float, default=0.2, help="clip parameter for PPO")
parser.add_argument('--random_seed', type=int, default=2021, help="number of random seeding")

# evaluation
parser.add_argument('--solved_reward', type=int, default=1000000, help="save the policy if avg_reward > solved_reward")
parser.add_argument('--log_interval', type=int, default=1, help="print avg reward in the interval")
parser.add_argument('--max_episodes', type=int, default=100, help="max training episodes")
parser.add_argument('--max_timesteps', type=int, default=500000000, help="max timesteps in one episode")
parser.add_argument('--render', type=bool, default=False, help="render")

args = parser.parse_args()

'''
# policy folders
folder_name = "PPO_switchsteering"
folder_name2 = "SolvedPPO_switchsteering"

try:
    os.mkdir(folder_name)
except OSError as e:
    print(e)

try:
    os.mkdir(folder_name2)
except OSError as e:
    print(e)
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ----------------------- PyTorch Implementation of PPO -----------------------
class Memory:
    """
    A memory space for PPO agent to pool the collected experience

    Args:
        actions (list): The actions that the agent took.
        states (list): The states that the agent observed.
        logprobs (list): The log probabilities of the chosen action.
        rewards (list): The rewards that the agent received from the environment after taking an action.
        is_terminals (list): The episode-terminal flag that the agent received after taking an action. Done or terminal.
    """
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    """
    Model (network) class, using actor-critic approach `actor-critic <https://arxiv.org/abs/1602.01783>`

    Actor-Critic has two models: the Actor and the Critic.

    The Actor corresponds to the policy π and it chooses the action and update the policy network.
    The actor network returns a probability for each action.

    The critic network estimates the value of the state-action pair. Based on these value functions, the critic evaluates the actions made by the actor.
    The critic network returns the estimated value of each action, given a state

    Args:
        state_dim (int): state space dimension
        action_dim (int): agent's action space dimension
        n_latent_var (int): number of variables in hidden layer
    """
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        """
        Passes the state observed into action_layer (actor network) to determine the action
        that the agent should take.
        
        Args:
            state (list): a list contatining the state observations

        Returns: 
            action (int): a number that indicates the action to be taken for gym environment
            log_prob (tensor): a tensor that contains the log probability of the action taken
        """
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    """
    Main PPO Class `PPO OpenAI <https://openai.com/blog/openai-baselines-ppo/>` `PPO paper <https://arxiv.org/abs/1707.06347>`

    Args:
        lr (float): learning rate
        betas (tuple): beta factor
        gamma (float): discount factor
        eps_clip (float): clip parameter for PPO
        K_epochs (int): update policy for K epochs
    """
    def __init__(self,
                 state_dim, action_dim, n_latent_var, lr,
                 betas, gamma, K_epochs, eps_clip
                 ):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(
            state_dim, action_dim, n_latent_var).to(device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(
            state_dim, action_dim, n_latent_var).to(device)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        """
        This function updates PPO Policy. 

        Monte Carlo estimation is used to estimate state rewards. Then the rewards are normalized and converted to tensor.
        Then the Policy will be optimized for K epochs by Evaluating old actions and values, Finding Surrogate Loss and taking gradient step.
        See: `PPO paper <https://arxiv.org/abs/1707.06347>`

        And at the end the weights will be updated.
        """
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(memory.rewards), reversed(memory.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


# -------------------------------- train loop ---------------------------------
def main():
    """
    Main train loop. 
    """
    
    env_name = 'snake-v0'
    env = gym.make(env_name)

    #env.make_window()
    state_dim = env.observation_space.shape[0]          # state space dimention
    print(state_dim)
    action_dim = env.action_space.n           # agent's action space dimention
    print(action_dim)

    # Load Model
    memory = Memory()
    ppo = PPO(state_dim, action_dim, args.n_latent_var,
              args.lr, args.betas, args.gamma, args.K_epochs, args.eps_clip)
    print(args.lr, args.betas)

    # log files
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    # create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"
    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # logging variables
    running_score = 0
    avg_length = 0
    timestep = 0
    rewardList = []

    # train loop
    for i_episode in range(1, args.max_episodes+1):
        state = env.reset()
        
        for t in range(args.max_timesteps):
            timestep += 1

            # Running policy_old:
            action_index = ppo.policy_old.act(state, memory)
            action = [0,0,0]
            action[action_index]=1
            print(action)
            state, reward, done, info = env.step(action)
            print(state, reward, done, info)

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % args.update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            running_score += reward
            if args.render:
                env.render()
            if done:
                break

        avg_length += t
        rewardList.append(running_score)
        #precision.append(correct_deliveries/total_deliveries)

        # save every 100 episodes
        '''if i_episode % 100 == 0:
            pre_name = f'100_switchsteering{i_episode}'
            torch.save(ppo.policy.state_dict(), os.path.join(
                folder_name, f'{pre_name}.pth'))'''

        # save the policy if avg_reward > solved_reward
        if running_score >= (args.log_interval*args.solved_reward):
            pre_name = f'Solved_switchsteering{i_episode}_{running_score}'
            torch.save(ppo.policy.state_dict(), os.path.join(f'{pre_name}.pth'))
            # break

        # logging
        if i_episode % args.log_interval == 0:

            print('Episode {} \t reward: {}'.format(i_episode, running_score))

            # log in logging file
            log_f.write('{},{},{}\n'.format(
                i_episode, avg_length, running_score))
            log_f.flush()

            running_score = 0


        # stop training if i_episode >= max_episodes/ plot results
        if i_episode >= args.max_episodes:
            print("####### END #######")
            pre_name = f'Solved_switchsteering{i_episode}_{running_score}'

            plt.plot(running_score, label='PPO')
            plt.ylabel('Precision')
            plt.xlabel('Episode')
            plt.legend(loc=0)
            plt.show()
            torch.save(ppo.policy.state_dict(), os.path.join(
                folder_name2, f'{pre_name}.pth'))
            break


class Agent_PPO:
    def train_step(self, world_env, counter_games):
        if(done):
            world_env.reset()
            
        return reward, done, score


# TODO Remove
if __name__ == '__main__':
    main()
