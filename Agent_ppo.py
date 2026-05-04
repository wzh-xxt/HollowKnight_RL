# -*- coding: utf-8 -*-
import numpy as np
import torch
class Agent:
    def __init__(self, act_dim, algorithm):
        self.act_dim = act_dim
        self.algorithm = algorithm

    def sample(self, station, soul, pos, train=None):
        prob_act, prob_move, value = self.algorithm.get_action_and_value(station, pos)
        if not train:

            act_tensor = prob_act.sample()
            move_tensor = prob_move.sample()

            act = act_tensor.item()
            move = move_tensor.item()

            if soul < 33 and act in [1, 2]:
                probs = prob_act.probs.clone()
                probs[0, 1] = 0
                probs[0, 2] = 0

                probs = probs / probs.sum()

                prob_act = torch.distributions.Categorical(probs=probs)
                act_tensor = prob_act.sample()
                act = act_tensor.item()

            return (move, act,value,prob_act.log_prob(act_tensor), prob_act.entropy(),
                    prob_move.log_prob(move_tensor), prob_move.entropy())

        else:
            return None, None, value, prob_act, prob_act.entropy(), prob_move, prob_move.entropy()


