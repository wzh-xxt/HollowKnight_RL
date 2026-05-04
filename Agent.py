# -*- coding: utf-8 -*-
import numpy as np

class Agent:
    def __init__(self,act_dim,algorithm,e_greed=0.42,e_greed_decrement=6.5e-5):
        self.act_dim = act_dim
        self.algorithm = algorithm
        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement


    def sample(self, station, soul,pos):
        pos = pos.unsqueeze(0)
        pred_move, pred_act = self.algorithm.model.predict(station,pos)
        pred_move = pred_move.cpu().numpy()
        pred_act = pred_act.cpu().numpy()
        sample = np.random.rand()  
        if sample < self.e_greed:
            move = np.random.randint(8)  # 随机探索wwkwwk
        else:
            move = np.argmax(pred_move)
        self.e_greed = max(
            0.05, self.e_greed - self.e_greed_decrement)

        sample = np.random.rand() 
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 随机探索
        else:
            act = np.argmax(pred_act)
            if soul < 33:
                if act == 1 or act == 2:
                    pred_act[0][1] = -30
                    pred_act[0][2] = -30
            act = np.argmax(pred_act)

        self.e_greed = max(
            0.05, self.e_greed - self.e_greed_decrement)
        return move, act,self.e_greed

                
