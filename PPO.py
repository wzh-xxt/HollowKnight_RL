import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPO:
    def __init__(self, model, learning_rate=1e-4):
        self.model = model

        self.act_dim = model.act_dim

        self.act_model = model.private_act_model
        self.move_model = model.private_move_model
        self.critic_model = model.critic_model

        self.shared_model = model.shared_model

        self.lr = learning_rate

        # self.optimizer = optim.Adam(list(self.shared_model.parameters())+
        #                             list(self.move_model.parameters())+list(self.act_model.parameters())+
        #                             list(self.critic_model.parameters()), lr=self.lr)
        self.optimizer = optim.Adam([
            {"params": self.shared_model.parameters(), "lr": self.lr / 3},
            {"params": self.act_model.parameters(), "lr": self.lr},
            {"params": self.move_model.parameters(), "lr": self.lr},
            {"params": self.critic_model.parameters(), "lr": self.lr / 3},
        ])

    def learn(self,loss, params,max_grad_norm):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, max_grad_norm)
        self.optimizer.step()

    def get_value(self, x, pos):
        return self.critic_model(self.shared_model(x),pos)

    def get_action_and_value(self, x,pos):
        hidden = self.shared_model(x)
        action = self.act_model(hidden,pos)
        move = self.move_model(hidden,pos)
        act_probs = Categorical(logits=action)
        move_probs = Categorical(logits=move)

        return act_probs,move_probs,self.critic_model(hidden,pos)
