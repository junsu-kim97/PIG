import torch.nn.functional as F
import sys

sys.path.append("../")
from models.distance import *
import numpy as np

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.
"""


def initialize_metrics(metric, dim):
    if metric == "L1":
        return L1()
    elif metric == "L2":
        return L2()
    elif metric == "dot":
        return DotProd()
    elif metric == "MLP":
        return MLPDist(dim)
    else:
        raise NotImplementedError


# define the actor network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params["action_max"]
        self.fc1 = nn.Linear(env_params["obs"] + env_params["goal"], 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, 400)
        self.action_out = nn.Linear(400, env_params["action"])

    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions


class critic(nn.Module):
    def __init__(self, env_params, args):
        super(critic, self).__init__()
        self.max_action = env_params["action_max"]
        self.inp_dim = env_params["obs"] + env_params["action"] + env_params["goal"]
        self.out_dim = 1
        self.mid_dim = 400

        if args.layer == 1:
            models = [nn.Linear(self.inp_dim, self.out_dim)]
        else:
            models = [nn.Linear(self.inp_dim, self.mid_dim)]
        if args.layer > 2:
            for i in range(args.layer - 2):
                models += [nn.ReLU(), nn.Linear(self.mid_dim, self.mid_dim)]
        if args.layer > 1:
            models += [nn.ReLU(), nn.Linear(self.mid_dim, self.out_dim)]

        self.base = nn.Sequential(*models)

    def forward(self, obs, goal, actions):
        x = torch.cat([obs, actions / self.max_action], dim=1)
        x = torch.cat([x, goal], dim=1)
        dist = self.base(x)
        return dist


class criticWrapper(nn.Module):
    def __init__(self, env_params, args):
        super(criticWrapper, self).__init__()
        self.base = critic(env_params, args)
        self.args = args
        self.gamma = args.gamma

    def forward(self, obs, goal, actions):
        dist = self.base(obs, goal, actions)
        self.alpha = np.log(self.gamma)
        return -(1 - torch.exp(dist * self.alpha)) / (1 - self.gamma)


class EmbedNet(nn.Module):
    def __init__(self, env_params, args):
        super(EmbedNet, self).__init__()
        self.max_action = env_params["action_max"]
        self.obs_dim = env_params["obs"] + env_params["action"]
        self.goal_dim = env_params["goal"]
        self.out_dim = 128
        self.mid_dim = 400

        if args.layer == 1:
            obs_models = [nn.Linear(self.obs_dim, self.out_dim)]
            goal_models = [nn.Linear(self.goal_dim, self.out_dim)]
        else:
            obs_models = [nn.Linear(self.obs_dim, self.mid_dim)]
            goal_models = [nn.Linear(self.goal_dim, self.mid_dim)]
        if args.layer > 2:
            for i in range(args.layer - 2):
                obs_models += [nn.ReLU(), nn.Linear(self.mid_dim, self.mid_dim)]
                goal_models += [nn.ReLU(), nn.Linear(self.mid_dim, self.mid_dim)]
        if args.layer > 1:
            obs_models += [nn.ReLU(), nn.Linear(self.mid_dim, self.out_dim)]
            goal_models += [nn.ReLU(), nn.Linear(self.mid_dim, self.out_dim)]

        self.obs_encoder = nn.Sequential(*obs_models)
        self.goal_encoder = nn.Sequential(*goal_models)
        self.metric = initialize_metrics(args.metric, self.out_dim)

    def forward(self, obs, goal, actions):
        s = torch.cat([obs, actions / self.max_action], dim=1)
        s = self.obs_encoder(s)
        g = self.goal_encoder(goal)
        dist = self.metric(s, g)
        return dist


class Qnet(nn.Module):
    def __init__(self, env_params, args):
        super(Qnet, self).__init__()
        self.mid_dim = 16
        self.metric = args.metric

        self.action_n = env_params["action_dim"]
        self.obs_fc1 = nn.Linear(env_params["obs"], 256)
        self.obs_fc2 = nn.Linear(256, self.mid_dim * self.action_n)

        self.goal_fc1 = nn.Linear(env_params["goal"], 256)
        self.goal_fc2 = nn.Linear(256, self.mid_dim)
        if self.metric == "MLP":
            self.mlp = nn.Sequential(
                nn.Linear(self.mid_dim * (self.action_n + 1), 128),
                nn.ReLU(),
                nn.Linear(128, self.action_n),
            )

    def forward(self, obs, goal):
        s = F.relu(self.obs_fc1(obs))
        s = F.relu(self.obs_fc2(s))
        s = s.view(s.size(0), self.action_n, self.mid_dim)

        g = F.relu(self.goal_fc1(goal))
        g = F.relu(self.goal_fc2(g))

        if self.metric == "L1":
            dist = torch.abs(s - g[:, None, :]).sum(dim=2)
        elif self.metric == "dot":
            dist = -(s * g[:, None, :]).sum(dim=2)
        elif self.metric == "L2":
            dist = ((torch.abs(s - g[:, None, :]) ** 2).sum(dim=2) + 1e-14) ** 0.5
        elif self.metric == "MLP":
            s = s.view(s.size(0), -1)
            x = torch.cat([s, g], dim=1)
            dist = self.mlp(x)
        else:
            raise NotImplementedError
        return dist


class QNetWrapper(nn.Module):
    def __init__(self, env_params, args):
        super(QNetWrapper, self).__init__()
        self.base = Qnet(env_params, args)
        self.args = args
        self.gamma = args.gamma

    def forward(self, obs, goal):
        dist = self.base(obs, goal)
        self.alpha = np.log(self.gamma)
        qval = -(1 - torch.exp(dist * self.alpha)) / (1 - self.gamma)
        return qval


class EmbedNetWrapper(nn.Module):
    def __init__(self, env_params, args):
        super(EmbedNetWrapper, self).__init__()
        self.base = EmbedNet(env_params, args)
        self.args = args
        self.gamma = args.gamma

    def forward(self, obs, goal, actions):
        dist = self.base(obs, goal, actions)
        self.alpha = np.log(self.gamma)
        return -(1 - torch.exp(dist * self.alpha)) / (1 - self.gamma)
