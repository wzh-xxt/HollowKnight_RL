# -*- coding: utf-8 -*-
import numpy as np
import torch
import time
import collections
from Model_ppo import Model
from PPO import PPO
from Agent_ppo import Agent
from Tool.tensor import to_tensor
import random
from torch.utils.tensorboard import SummaryWriter
import Tool.Helper
import Tool.Actions
from Tool.Helper import mean, is_end
from Tool.Actions import take_action, restart, take_direction, TackAction, restart1, Nothing
from Tool.WindowsAPI import grab_screen
from Tool.GetHP import Hp_getter
from Tool.FrameBuffer import FrameBuffer

window_size = (0, 0, 1920, 1017)
station_size = (162, 166, 1762, 966)

HP_WIDTH = 768
HP_HEIGHT = 407
WIDTH = 400
HEIGHT = 200
ACTION_DIM = 5
FRAMEBUFFERSIZE = 4
INPUT_SHAPE = (FRAMEBUFFERSIZE, HEIGHT, WIDTH, 3)

action_name = ["Attack", "Attack_Up",
               "Short_Jump", "Mid_Jump", "Skill_Up",
               "Skill_Down", "Rush", "Cure"]

move_name = ["Move_Left", "Move_Right", "Turn_Left", "Turn_Right"]

DELAY_REWARD = 1
LEARNING_RATE = 1.5e-4  # 学习率


def reset():
    Nothing()
    time.sleep(1)
    restart()
    print(0)

    time.sleep(5.5)
    while True:
        boss_hp_value = hp.get_boss_hp()
        self_hp = hp.get_self_hp()
        boss_x = hp.get_hornet_location()
        self_x = hp.get_play_location()
        print(boss_hp_value, self_hp, boss_x, self_x)
        if self_hp >= 0 and self_hp <= 8 and boss_hp_value > 900:
            time.sleep(15)
            restart1()
            time.sleep(5)
            boss_hp_value = hp.get_boss_hp()
            self_hp = hp.get_self_hp()
        if boss_hp_value >= 800 and boss_hp_value <= 900 and self_hp >= 1 and self_hp <= 8:
            break
        time.sleep(45)
        if boss_hp_value >= 800 and boss_hp_value <= 900 and self_hp == 0:
            restart()
            time.sleep(3)
        if boss_x == 0:
            restart()
            time.sleep(3)
        if self_hp >= 0 and self_hp <= 8 and boss_hp_value == 0:
            restart()
            time.sleep(3)
        if self_hp >= 0 and self_hp <= 8 and boss_hp_value > 900:
            restart1()
            time.sleep(3)
        if self_hp >= 0 and self_hp <= 8 and boss_hp_value < 800:
            restart()
            time.sleep(3)
        if self_hp == 0 and boss_hp_value == 0:
            restart()
            time.sleep(3)


def run_episode(hp, agent, PASS_COUNT, device, move_deque,self_x_deque):
    reset()
    step = 0
    total_act = 0
    total_move = 0
    num_steps = 352
    update_epoch = 5
    minibatch_size= 32
    clip_coef = 0.2
    norm_adv = True
    clip_vloss = True
    ent_coef = 0.005
    vf_coef = 0.15
    act_coef = 0.75
    move_coef = 1.5
    max_grad_norm = 0.5
    gae_lambda = 0.95
    GAMMA = 0.94

    thread = FrameBuffer(1, "FrameBuffer", WIDTH, HEIGHT, maxlen=FRAMEBUFFERSIZE)
    thread.start()

    obs = np.zeros((num_steps,) + INPUT_SHAPE, dtype=np.float32)
    actions = np.zeros(num_steps, dtype=np.int64)
    moves = np.zeros(num_steps, dtype=np.int64)
    act_reward = np.zeros(num_steps, dtype=np.float32)
    move_reward = np.zeros(num_steps, dtype=np.float32)
    pos = np.zeros((num_steps, 3), dtype=np.float32)
    dones = np.zeros(num_steps, dtype=np.float32)

    logprobs_act = torch.zeros(num_steps).to(device)
    logprobs_move = torch.zeros(num_steps).to(device)
    values = torch.zeros(num_steps,2).to(device)
    act_advantages = torch.zeros(num_steps).to(device)
    move_advantages = torch.zeros(num_steps).to(device)
    while (len(thread.buffer) < FRAMEBUFFERSIZE):
        time.sleep(0.01)
    next_obs = thread.get_buffer()
    next_done = 0
    next_boss_hp_value = hp.get_boss_hp()
    next_self_hp = hp.get_self_hp()
    next_boss_x = hp.get_hornet_location()
    next_self_x = hp.get_play_location()
    next_soul = hp.get_souls()
    next_pos = [next_self_x / 35, next_boss_x / 35, next_boss_x / 35 - next_self_x / 35]

    for step in range(0, num_steps):

        obs[step] = next_obs
        dones[step] = next_done
        obs_tensor = to_tensor(obs[step]).to(device)  # 加batch维
        boss_hp_value = next_boss_hp_value
        self_hp = next_self_hp
        boss_x = next_boss_x
        self_x = next_self_x
        soul = next_soul
        pos[step] =  next_pos
        pos_tensor = torch.tensor(next_pos, dtype=torch.float32).to(device)

        with torch.no_grad():
            move, action,value_pred,logprob_act,_,logprob_move,_ = agent.sample(obs_tensor, soul, pos_tensor.unsqueeze(0))

        actions[step] = action
        moves[step] = move
        logprobs_act[step] = logprob_act
        logprobs_move[step] = logprob_move
        values[step] = value_pred

        move_deque.append(move)
        self_x_deque.append(self_x)

        print("攻击:", action, "移动：", move)

        #执行行动
        take_direction(move)
        take_action(action)

        while (len(thread.buffer) < FRAMEBUFFERSIZE):
            time.sleep(0.05)
        next_obs = thread.get_buffer()
        next_boss_hp_value = hp.get_boss_hp()
        next_self_hp = hp.get_self_hp()
        next_boss_x = hp.get_hornet_location()
        next_self_x = hp.get_play_location()
        next_soul = hp.get_souls()
        next_pos = [next_self_x / 35, next_boss_x / 35, next_boss_x / 35 - next_self_x / 35]

        # get reward
        move_reward[step], next_done = Tool.Helper.move_judge(boss_hp_value, next_boss_hp_value, self_hp, next_self_hp, self_x,
                                                   next_self_x,
                                                   boss_x, next_boss_x, move, move_deque,self_x_deque)
        act_reward[step], next_done = Tool.Helper.action_judge(boss_hp_value, next_boss_hp_value, self_hp, next_self_hp, self_x,
                                                    next_self_x,
                                                    boss_x, next_boss_x, action)
        total_act += act_reward[step]
        total_move += move_reward[step]

        if next_done == 1:
            Tool.Actions.Nothing()
            reset()
    Nothing()

    with torch.no_grad():
        next_value = agent.algorithm.get_value(to_tensor(next_obs).to(device),torch.tensor(next_pos).unsqueeze(0).to(device)).reshape(1, 2)
        lastgaelam_act = 0
        lastgaelam_move = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1].view(1, 2)
            # TD errorls
            act_delta = act_reward[t] + GAMMA * nextvalues[0][0] * nextnonterminal - values[t][0]
            move_delta = move_reward[t] + GAMMA * nextvalues[0][1] * nextnonterminal - values[t][1]
            # A[t]就是Q-V,只不过应为只是ppo，所以要累加，每一个t都参与训练,用来参与训练policy
            act_advantages[t] = lastgaelam_act = act_delta  + GAMMA * gae_lambda * nextnonterminal * lastgaelam_act
            move_advantages[t] = lastgaelam_move = move_delta + GAMMA * gae_lambda * nextnonterminal * lastgaelam_move
        # Advantage = Q − V，所以Q ≈ Return = Advantage + V，用来训练价值网络。
        act_returns = act_advantages + values[:, 0]
        move_returns = move_advantages + values[:, 1]
        # act_returns = (act_returns - act_returns.mean()) / (act_returns.std() + 1e-8)
        # move_returns = (move_returns - move_returns.mean()) / (move_returns.std() + 1e-8)

    # Optimizing the policy and value network
    b_inds = np.arange(num_steps)
    obs_tensor = to_tensor(obs).to(device)
    pos_tensor = to_tensor(pos).to(device)
    loss_act,loss_move,loss_value,loss_entropy = 0,0,0,0
    for epoch in range(update_epoch):
        print("开始训练")
        np.random.shuffle(b_inds)
        for start in range(0, num_steps, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, _, new_values, newlogprob_act, act_entropy, newlogprob_move, move_entropy= agent.sample(obs_tensor[mb_inds],
                                                                                                           soul,pos_tensor[mb_inds],True)

            newlogprob_act = newlogprob_act.log_prob(torch.tensor(actions[mb_inds]).to(device))
            newlogprob_move = newlogprob_move.log_prob(torch.tensor(moves[mb_inds]).to(device))

            act_logratio = newlogprob_act - logprobs_act[mb_inds]
            move_logratio = newlogprob_move - logprobs_move[mb_inds]

            act_ratio = act_logratio.exp()
            move_ratio = move_logratio.exp()

            mbact_advantages = act_advantages[mb_inds]
            mbmove_advantages = move_advantages[mb_inds]
            # if norm_adv:
            #     mbact_advantages = (mbact_advantages - mbact_advantages.mean()) / (mbact_advantages.std() + 1e-8)
            #     mbmove_advantages = (mbmove_advantages - mbmove_advantages.mean()) / (mbmove_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mbact_advantages * act_ratio
            pg_loss2 = -mbact_advantages * torch.clamp(act_ratio, 1 - clip_coef, 1 + clip_coef)
            act_loss = torch.max(pg_loss1, pg_loss2).mean()
            pg_loss1 = -mbmove_advantages * move_ratio
            pg_loss2 = -mbmove_advantages * torch.clamp(move_ratio, 1 - clip_coef, 1 + clip_coef)
            move_loss = torch.max(pg_loss1, pg_loss2).mean()
            pg_loss = act_coef * act_loss + move_coef * move_loss

            # Value loss
            newvalue = new_values
            returns = torch.stack([act_returns, move_returns], dim=1)
            if clip_vloss:
                v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                v_clipped = values[mb_inds] + torch.clamp(
                    newvalue - values[mb_inds],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

            entropy_loss = act_entropy.mean() / 2 + move_entropy.mean() / 2

            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

            agent.algorithm.learn(loss,agent.algorithm.model.parameters(),max_grad_norm)

            loss_act += act_loss.item() / update_epoch / (num_steps/minibatch_size)
            loss_move += move_loss.item() / update_epoch / (num_steps/minibatch_size)
            loss_value += (v_loss * vf_coef).item() / update_epoch / (num_steps/minibatch_size)
            loss_entropy +=(ent_coef * entropy_loss).item() / update_epoch / (num_steps/minibatch_size)

    thread.stop()
    # if episode==1 or episode==2 or episode==3:
    #     time.sleep(5.5)
    return total_act, total_move, step, PASS_COUNT, self_hp,loss_act,loss_move,loss_value,loss_entropy


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_remind_hp = 0
    writer = SummaryWriter("runs/hollow_knight_dqn")
    # new model, if exit save file, load itk
    model = Model(INPUT_SHAPE, ACTION_DIM)
    model.to(device)
    move_deque = collections.deque(maxlen=6)
    self_x_deque = collections.deque(maxlen=3)
    # Hp counter
    hp = Hp_getter()

    model.load_model()
    algorithm = PPO(model, learning_rate=LEARNING_RATE)
    agent = Agent(ACTION_DIM, algorithm)

    max_episode = 30000
    # 开始训练
    episode = 0
    PASS_COUNT = 0
    reward_history = []  # pass count
    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # 训练
        episode += 1

        total_act, total_move, total_step, PASS_COUNT, remind_hp,loss_act,loss_move,loss_value,loss_entropy = run_episode(hp, agent, PASS_COUNT,device, move_deque,self_x_deque)
        writer.add_scalar("Act_Reward", total_act, episode)
        writer.add_scalar("Move_Reward", total_move, episode)
        writer.add_scalar("loss_act", loss_act, episode)
        writer.add_scalar("loss_move", loss_move, episode)
        writer.add_scalar("loss_value", loss_value, episode)
        writer.add_scalar("loss_entropy", loss_entropy, episode)
        writer.add_scalar("remain_hp", remind_hp, episode)
        writer.add_scalar("steps", total_step, episode)
        if episode % 5 == 1:
            model.save_model(episode)
        total_remind_hp += remind_hp
        print("Episode: ", episode, ",Act_reward: ", total_act, ",Move_reward: ", total_move,
              ",loss_act: ", loss_act, ",loss_move: ", loss_move,",loss_value: ", loss_value, ",loss_entropy: ", loss_entropy,
              ", pass_count: ", PASS_COUNT, ", hp:", total_remind_hp / episode)
