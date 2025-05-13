import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        #self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        #self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.feature_layer = FeatureLayer(input_shape, args.rnn_hidden_dim)
        if self.args.n_head_layer==2:
            self.head = Head(args.rnn_hidden_dim, args.n_actions, hidden_dim=args.rnn_hidden_dim)
        else:
            self.head = Head(args.rnn_hidden_dim, args.n_actions)
        self.head_dict = {}

    def init_hidden(self):
        # make hidden states on same device as model
        return self.feature_layer.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, head_id=None):
        """x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)"""
        h = self.feature_layer(inputs, hidden_state)
        if head_id is not None and head_id in self.head_dict: # if not in head_dict, the newest one
            q = self.head_dict[head_id](h)
        else:
            q = self.head(h)
        return q, h

    def cache_head(self, head_id):
        self.head_dict[head_id] = copy.deepcopy(self.head)

    def reset_head(self):
        reset_recursive(self.head)

    def load_head(self, head_id):
        self.head.load_state_dict(self.head_dict[head_id].state_dict())
    
    def reset_param(self):
        reset_recursive(self.feature_layer)
        reset_recursive(self.head)
    
class FeatureLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=None) -> None:
        super(FeatureLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        return h

class Head(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None) -> None:
        super(Head, self).__init__()
        if hidden_dim:
            self.layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)

def reset_recursive(x):
    if len(list(x.children())) == 0:
        if hasattr(x, 'reset_parameters'):
            x.reset_parameters()
        return
    for layer in x.children():
        reset_recursive(layer)
