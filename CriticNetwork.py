import torch
import torch.nn as nn


class CriticNetwork(nn.Module):
    def __init__(self, observation_space_size):
        super(CriticNetwork, self).__init__()
        self.hidden1 = nn.Linear(in_features=observation_space_size, out_features=128)
        self.hidden2 = nn.Linear(in_features=128, out_features=128)
        self.hidden3 = nn.Linear(in_features=128, out_features=256)
        self.hidden4 = nn.Linear(in_features=256, out_features=128)
        self.hidden1_act = nn.ReLU()
        self.hidden2_act = nn.ReLU()
        self.hidden3_act = nn.ReLU()
        self.hidden4_act = nn.ReLU()

        self.value_estimation = nn.Linear(in_features=128, out_features=1)
        pass

    def forward(self, state):
        x = self.hidden1_act(self.hidden1(state))
        x = self.hidden2_act(self.hidden2(x))
        x = self.hidden3_act(self.hidden3(x))
        x = self.hidden4_act(self.hidden4(x))

        value = self.value_estimation(x)
        return value