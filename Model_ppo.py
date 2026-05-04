import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.distributions.categorical import Categorical


# ---------------- BasicBlock ----------------
class BasicBlock(nn.Module):
    def __init__(self, in_num, out_num,stride=1):
        super(BasicBlock, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv3d(in_num, out_num, kernel_size=3,stride=stride, padding=1)
        self.conv2 = nn.Conv3d(out_num, out_num, kernel_size=3,stride=1, padding=1)

        if stride != 1 or in_num != out_num:
            self.downsample = nn.Conv3d(in_num, out_num,kernel_size=1, stride=stride)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = F.relu(out)

        out = self.conv2(out)

        out = out + identity
        out = F.relu(out)
        return out


# ---------------- Model ----------------
class Model(nn.Module):
    def __init__(self, input_shape, act_dim):
        super(Model, self).__init__()

        self.act_dim = act_dim

        # -------- shared --------
        self.shared_model = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(2,3,3), stride=(1,2,2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2), stride=1),

            self.build_resblock(32,64),
            self.build_resblock(64, 128, stride=2),
            self.build_resblock(128, 256, stride=2),
        )

        # -------- act head --------
        self.private_act_model = ActModel(act_dim)

        # -------- move head --------
        self.private_move_model = MoveModel(7)

        self.critic_model = Critic()


    def build_resblock(self, in_num,out_num, stride=1):
        layers = []
        layers.append(BasicBlock(in_num,out_num, stride))

        layers.append(BasicBlock(out_num,out_num, stride=1))

        return nn.Sequential(*layers)


    def forward(self, x,pos):
        feat = self.shared_model(x)

        act = self.private_act_model(feat,pos)
        move = self.private_move_model(feat,pos)
        value = self.critic(feat,pos)

        return act,move,value


    def predict(self, x,pos):

        move, act, value = self.forward(x,pos)
        return move, act,value

    def save_model(self,i, path="./model"):
        os.makedirs(path, exist_ok=True)
        print("save model")
        torch.save({
            "shared": self.shared_model.state_dict(),
            "act": self.private_act_model.state_dict(),
            "move": self.private_move_model.state_dict(),
            "critic": self.critic_model.state_dict(),
        }, os.path.join(path, f"model{i}.pth"))

    def load_model(self, path="./model"):
        model_path = os.path.join(path, "model71.pth")

        if os.path.exists(model_path):
            print("load model")

            checkpoint = torch.load(model_path, map_location="cpu")

            self.shared_model.load_state_dict(checkpoint["shared"])
            self.private_act_model.load_state_dict(checkpoint["act"])
            self.private_move_model.load_state_dict(checkpoint["move"])
            self.critic_model.load_state_dict(checkpoint["critic"])


        else:
            print("no saved model found")

class ActModel(nn.Module):
    def __init__(self, act_dim):
        super(ActModel, self).__init__()
        self.conv = self.build_resblock(256, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.act_dim=act_dim

        self.pos_fc = nn.Sequential(nn.Linear(3, 16),nn.ReLU(),nn.Linear(16, 16))
        self.head = nn.Sequential(
            nn.Linear(256 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, feat, pos):
        feat = self.conv(feat)
        feat = self.pool(feat)
        feat = feat.view(feat.size(0), -1)

        pos_feat = self.pos_fc(pos)


        x = torch.cat([feat, pos_feat], dim=1)

        q_values = self.head(x)
        return q_values

    def build_resblock(self, in_num,out_num, stride=1):
        layers = []
        layers.append(BasicBlock(in_num,out_num, stride))

        layers.append(BasicBlock(out_num,out_num, stride=1))

        return nn.Sequential(*layers)

class MoveModel(nn.Module):
    def __init__(self,move_dim):
        super(MoveModel, self).__init__()
        self.conv = self.build_resblock(256, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.move_dim = move_dim

        self.pos_fc = nn.Sequential(nn.Linear(3, 16),nn.ReLU(),nn.Linear(16, 16))
        self.head = nn.Sequential(
            nn.Linear(256 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, move_dim)
        )

    def forward(self, feat, pos):
        feat = self.conv(feat)
        feat = self.pool(feat)
        feat = feat.view(feat.size(0), -1)

        pos_feat = self.pos_fc(pos)

        x = torch.cat([feat, pos_feat], dim=1)

        q_values = self.head(x)
        return q_values

    def build_resblock(self, in_num,out_num, stride=1):
        layers = []
        layers.append(BasicBlock(in_num,out_num, stride))

        layers.append(BasicBlock(out_num,out_num, stride=1))

        return nn.Sequential(*layers)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv = self.build_resblock(256, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.pos_fc = nn.Sequential(nn.Linear(3, 16),nn.ReLU(),nn.Linear(16, 16))
        self.head1 = nn.Sequential(
            nn.Linear(256 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.head2 = nn.Sequential(
            nn.Linear(256+16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, feat, pos):
        feat = self.conv(feat)
        feat = self.pool(feat)
        act = feat.view(feat.size(0), -1)
        pos_feat = self.pos_fc(pos)

        act = torch.cat([act, pos_feat], dim=1)

        act_values = self.head1(act)

        move = feat.view(feat.size(0), -1)

        move = torch.cat([move, pos_feat], dim=1)

        move_values = self.head2(move)
        values = torch.cat([act_values, move_values], dim=1)
        return values

    def build_resblock(self, in_num,out_num, stride=1):
        layers = []
        layers.append(BasicBlock(in_num,out_num, stride))

        layers.append(BasicBlock(out_num,out_num, stride=1))

        return nn.Sequential(*layers)

# class Critic(nn.Module):
#     def __init__(self):
#         super(Critic, self).__init__()
#         self.conv = self.build_resblock(256, 256, stride=2)
#         self.pool = nn.AdaptiveAvgPool3d(1)
#         self.pos_fc = nn.Sequential(nn.Linear(3, 16),nn.ReLU(),nn.Linear(16, 16))
#         self.head = nn.Sequential(
#             nn.Linear(256 + 16, 128),
#             nn.ReLU(),
#             nn.Linear(128, 2)
#         )
#
#     def forward(self, feat, pos):
#         feat = self.conv(feat)
#         feat = self.pool(feat)
#         feat = feat.view(feat.size(0), -1)
#         pos_feat = self.pos_fc(pos)
#
#         feat = torch.cat([feat, pos_feat], dim=1)
#
#         values = self.head(feat)
#         return values
#
#     def build_resblock(self, in_num,out_num, stride=1):
#         layers = []
#         layers.append(BasicBlock(in_num,out_num, stride))
#
#         layers.append(BasicBlock(out_num,out_num, stride=1))
#
#         return nn.Sequential(*layers)