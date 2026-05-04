import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class DQN:
    def __init__(self, model, gamma=0.9, learning_rate=1e-4):
        self.model = model

        self.act_dim = model.act_dim

        self.act_model = model.private_act_model
        self.move_model = model.private_move_model

        self.act_target_model = model.private_act_target_model
        self.move_target_model = model.private_move_target_model

        self.shared_model = model.shared_model
        self.shared_target_model = model.shared_target_model

        self.gamma = gamma
        self.lr = learning_rate

        # optimizer + loss
        # self.act_optimizer = optim.Adam(self.act_model.parameters(),lr=self.lr)
        #
        # self.move_optimizer = optim.Adam(self.move_model.parameters(),lr=self.lr)

        # self.shared_optimizer = optim.Adam(self.shared_model.parameters(), lr=self.lr)
        self.optimizer = optim.Adam(list(self.shared_model.parameters())+list(self.move_model.parameters())+list(self.act_model.parameters()), lr=self.lr)

        self.loss_func = nn.MSELoss()

        self.act_global_step = 0
        self.move_global_step = 0
        self.update_target_steps = 100


    # ---------------- act ----------------

    def act_train_step(self, action, obs, reward,next_obs,bacth_pos,batch_next_pos, terminal):

        feat = self.shared_model(obs)
        predictions = self.act_model(feat,bacth_pos)
        pred_action_value = predictions.gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            feat_target = self.shared_target_model(next_obs)
            predictions_target = self.act_target_model(feat_target,batch_next_pos,)
            pred_action_value_target = predictions_target.max(dim=1)[0]

        terminal = torch.tensor(terminal, dtype=torch.float32).to(obs.device)

        loss = self.loss_func(pred_action_value, reward+pred_action_value_target * self.gamma * (1.0 - terminal))

        return loss


    def act_train_model(self, action, obs, reward, next_obs, bacth_pos,batch_next_pos,terminal,epochs=1):
        for _ in range(epochs):
            loss = self.act_train_step(action, obs, reward,next_obs,bacth_pos, batch_next_pos,terminal)
        return loss


    def act_learn(self, obs, action, reward, next_obs,bacth_pos,batch_next_pos, terminal):
        loss = self.act_train_model(action, obs, reward,next_obs, bacth_pos,batch_next_pos,terminal, epochs=1)
        self.act_global_step += 1
        return loss



    def act_replace_target(self):

        self.act_target_model.load_state_dict(self.act_model.state_dict())


    # ---------------- move ----------------

    def move_train_step(self, action, obs, reward,next_obs, batch_pos,batch_next_pos,terminal):
        feat = self.shared_model(obs)
        predictions = self.move_model(feat,batch_pos)
        pred_action_value = predictions.gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            feat_target = self.shared_target_model(next_obs)
            predictions_target = self.move_target_model(feat_target,batch_next_pos,)
            pred_action_value_target = predictions_target.max(dim=1)[0]

        terminal = torch.tensor(terminal, dtype=torch.float32).to(obs.device)
        loss = self.loss_func(pred_action_value, reward + pred_action_value_target * self.gamma * (1.0 - terminal))

        return loss


    def move_train_model(self, action, obs, reward, next_obs, batch_pos,batch_next_pos,terminal,epochs=1):
        for _ in range(epochs):
            loss = self.move_train_step(action, obs, reward,next_obs,batch_pos,batch_next_pos, terminal)
        return loss


    def move_learn(self, obs, action, reward, next_obs,batch_pos,batch_next_pos, terminal):
        loss = self.move_train_model(action, obs, reward,next_obs,batch_pos, batch_next_pos,terminal, epochs=1)
        self.move_global_step += 1
        return loss


    def move_replace_target(self):
        self.move_target_model.load_state_dict(self.move_model.state_dict())

    def shared_learn(self,loss_move,loss_act):
        self.optimizer.zero_grad()
        loss = loss_act + loss_move
        loss.backward()
        self.optimizer.step()

    # ---------------- shared target ----------------
    def replace_target(self):
        tau = 0.003  # 可以调

        self.soft_update(self.shared_target_model, self.shared_model, tau)
        self.soft_update(self.act_target_model, self.act_model, tau)
        self.soft_update(self.move_target_model, self.move_model, tau)

    def soft_update(self, target_net, source_net, tau=0.003):
        with torch.no_grad():
            for target_param, param in zip(target_net.parameters(), source_net.parameters()):
                target_param.copy_(
                    tau * param + (1 - tau) * target_param
                )
