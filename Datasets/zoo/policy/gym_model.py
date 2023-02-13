import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from policy.base_model import BasePolicy


class GymPolicy(BasePolicy):
    def __init__(self, config, policy_id=-1):
        super(GymPolicy, self).__init__()
        self.policy_id = policy_id  # Parent policy when id = -1, Child policy id >= 0
        self.state_num = config['state_num']
        self.action_num = config['action_num']
        self.discrete_action = config['discrete_action']
        if "add_gru" in config:
            self.add_gru = config['add_gru']
        else:
            self.add_gru = True
        self.fc1 = nn.Linear(self.state_num, 32)
        if self.add_gru:
            self.gru = nn.GRU(32, 32, 1)  # input_size, input_size, num_layers; Shape: L, N, H_{input}
            self.h = torch.zeros([1, 1, 32], dtype=torch.float)  # Shape : D * num_layers, N, H_{out}
        self.fc2 = nn.Linear(32, self.action_num)

    def forward(self, x):
        with torch.no_grad():  # Will not call Tensor.backward()
            x = torch.from_numpy(x).float()
            x = x.unsqueeze(0)  # maybe equal to torch.unsqueeze(x, 0)
            x = torch.tanh(self.fc1(x))

            if self.add_gru:
                x, self.h = self.gru(x, self.h)
                x = torch.tanh(x)  # maybe remove it ? let us test it in the future

            x = self.fc2(x)

            if self.discrete_action:
                x = F.softmax(x.squeeze(), dim=0)  # all the dimensions of input of size 1 removed.
                x = torch.argmax(x)
            else:
                x = torch.tanh(x.squeeze())

            x = x.detach().cpu().numpy()
            return x

    def set_policy_id(self, policy_id):
        self.policy_id = policy_id

    def zero_init(self):
        for param in self.parameters():
            param.dataSetName = torch.zeros(param.shape)

    def norm_init(self, std=1.0):
        for param in self.parameters():
            shape = param.shape
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            param.dataSetName = torch.from_numpy(out)

    def reset(self):
        if self.add_gru:
            self.h = torch.zeros([1, 1, 32], dtype=torch.float)

    def set_policy_id(self, policy_id):
        self.policy_id = policy_id

    def get_param_list(self):
        param_lst = []
        for param in self.parameters():
            param_lst.append(param.data.numpy())
        return param_lst

    def set_param_list(self, param_lst: list):
        lst_idx = 0
        for param in self.parameters():
            param.dataSetName = torch.tensor(param_lst[lst_idx]).float()
            lst_idx += 1
