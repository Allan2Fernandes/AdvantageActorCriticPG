import torch
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork


class A2CAgent:
    def __init__(self, action_space_size, state_space_size, device):
        self.gamma = 0.90
        self.actor_network = ActorNetwork(action_space_size=action_space_size, observation_space_size=state_space_size).to(device=device)
        self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=0.0001)
        self.critic_network = CriticNetwork(observation_space_size=state_space_size).to(device=device)
        self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(), lr=0.0001)
        self.episode_rewards = []
        self.log_probs = []
        pass

    def act(self, state, device):
        means, devs = self.actor_network(state)
        action_distribution = torch.distributions.normal.Normal(means, devs)
        action = action_distribution.sample() # Exploration vs Exploitation is taken care of by the random nature of sampling a distribution
        action = torch.tensor([torch.squeeze(torch.squeeze(action, dim=0))])
        action = action.to(device)
        return action, action_distribution.log_prob(action)

    def train_both_networks(self, log_prob, reward, state, next_state, done):
        advantage = reward + (1.0-done)*self.gamma*self.critic_network(next_state) - self.critic_network(state)

        # Train the actor
        actor_loss = -log_prob*advantage.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Train the critic
        critic_loss = advantage.pow(2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pass
