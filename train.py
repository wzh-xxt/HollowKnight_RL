# -*- coding: utf-8 -*-
import numpy as np
import torch
import time
import collections

from Model import Model
from DQN import DQN
from Agent import Agent
from ReplayMemory import ReplayMemory
from Tool.tensor import to_tensor
import random
from torch.utils.tensorboard import SummaryWriter
import Tool.Helper
import Tool.Actions
from Tool.Helper import mean, is_end
from Tool.Actions import take_action, restart,take_direction, TackAction, restart1
from Tool.WindowsAPI import grab_screen
from Tool.GetHP import Hp_getter
from Tool.UserInput import User
from Tool.FrameBuffer import FrameBuffer

window_size = (0,0,1920,1017)
station_size = (162, 166, 1762, 966)

HP_WIDTH = 768
HP_HEIGHT = 407
WIDTH = 400
HEIGHT = 200
ACTION_DIM = 7
FRAMEBUFFERSIZE = 4
INPUT_SHAPE = (FRAMEBUFFERSIZE, HEIGHT, WIDTH, 3)

MEMORY_SIZE = 2250  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 24  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 20  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.00001 # 学习率
GAMMA = 0.9

action_name = ["Attack", "Attack_Up",
           "Short_Jump", "Mid_Jump", "Skill_Up", 
           "Skill_Down", "Rush", "Cure"]

move_name = ["Move_Left", "Move_Right", "Turn_Left", "Turn_Right"]

DELAY_REWARD = 1




def run_episode(hp, algorithm,agent,act_rmp_correct, move_rmp_correct,PASS_COUNT,paused,device,move_deque,episode):
    restart()
    print("训练")
    for i in range(2):
        print("训练中")
        if (len(move_rmp_correct) > MEMORY_WARMUP_SIZE):
            # print("move learning")
            batch_station,batch_actions,batch_reward,batch_next_station,batch_pos,batch_next_pos,batch_done = move_rmp_correct.sample(BATCH_SIZE)
            batch_station = to_tensor(batch_station)
            batch_next_station = to_tensor(batch_next_station)
            batch_actions = torch.tensor(batch_actions, dtype=torch.long).to(device)
            batch_reward = torch.tensor(batch_reward, dtype=torch.float32).to(device)
            batch_pos = torch.tensor(batch_pos, dtype=torch.float32).to(device)
            batch_next_pos = torch.tensor(batch_next_pos, dtype=torch.float32).to(device)

            loss_move = algorithm.move_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_pos,batch_next_pos,batch_done)

        if (len(act_rmp_correct) > MEMORY_WARMUP_SIZE):
            # print("action learning")wwk
            batch_station,batch_actions,batch_reward,batch_next_station,batch_pos,batch_next_pos,batch_done = act_rmp_correct.sample(BATCH_SIZE)
            batch_station = to_tensor(batch_station)
            batch_next_station = to_tensor(batch_next_station)
            batch_actions = torch.tensor(batch_actions, dtype=torch.long).to(device)
            batch_reward = torch.tensor(batch_reward, dtype=torch.float32).to(device)
            batch_pos = torch.tensor(batch_pos, dtype=torch.float32).to(device)
            batch_next_pos = torch.tensor(batch_next_pos, dtype=torch.float32).to(device)

            loss_act = algorithm.act_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_pos,batch_next_pos,batch_done)
        if (len(act_rmp_correct) > MEMORY_WARMUP_SIZE):
            algorithm.shared_learn(loss_move,loss_act)

    print("训练结束")

    
    step = 0
    done = 0
    total_act = 0
    total_move = 0

    start_time = time.time()
    # Delay Reward
    DelayMoveReward = collections.deque(maxlen=DELAY_REWARD)
    DelayActReward = collections.deque(maxlen=DELAY_REWARD)
    DelayStation = collections.deque(maxlen=DELAY_REWARD) # 1 more for next_station
    DelayActions = collections.deque(maxlen=DELAY_REWARD)
    DelayDirection = collections.deque(maxlen=DELAY_REWARD)
    DelayPosition = collections.deque(maxlen=DELAY_REWARD)
    DelayStation_next = collections.deque(maxlen=DELAY_REWARD)
    DelayPosition_next = collections.deque(maxlen=DELAY_REWARD)

    print(0)
    # if episode==1 or episode==2 or episode==3:
    #     time.sleep(5.5)

    while True:
        boss_hp_value = hp.get_boss_hp()
        self_hp = hp.get_self_hp()
        boss_x = hp.get_hornet_location()
        self_x = hp.get_play_location()
        print(boss_hp_value,self_hp,boss_x,self_x)
        if self_hp >=0 and  self_hp<= 8 and boss_hp_value>900:
            time.sleep(15)
            restart1()
            time.sleep(5)
            boss_hp_value = hp.get_boss_hp()
            self_hp = hp.get_self_hp()
        if boss_hp_value >=800 and  boss_hp_value <= 900 and self_hp >= 1 and self_hp <= 8:
            break
        time.sleep(45)
        if boss_hp_value >=800 and  boss_hp_value <= 900 and self_hp==0:
            restart()
            time.sleep(3)
        if boss_x==0:
            restart()
            time.sleep(3)
        if self_hp >=0 and  self_hp<= 8 and boss_hp_value==0:
            restart()
            time.sleep(3)
        if self_hp >=0 and  self_hp<= 8 and boss_hp_value>900:
            restart1()
            time.sleep(3)
        if self_hp >=0 and  self_hp<= 8 and boss_hp_value<800:
            restart()
            time.sleep(3)
        if self_hp==0 and boss_hp_value==0:
            restart()
            time.sleep(3)
    print(1)

    thread1 = FrameBuffer(1, "FrameBuffer", WIDTH, HEIGHT, maxlen=FRAMEBUFFERSIZE)
    thread1.start()

    print(2)

    # last_hornet_y = 0
    while True:
        step += 1
        
        while(len(thread1.buffer) < FRAMEBUFFERSIZE):
            time.sleep(0.1)
        
        stations = thread1.get_buffer()
        stations_tensor = to_tensor(stations)  # 加batch维
        boss_hp_value = hp.get_boss_hp()
        self_hp = hp.get_self_hp()
        boss_x = hp.get_hornet_location()
        self_x = hp.get_play_location()
        soul = hp.get_souls()
        pos = [self_x/35, boss_x/35, boss_x/35 - self_x/35]
        pos_tensor = torch.tensor(pos, dtype=torch.float32).to(device)

        move, action, greedy = agent.sample(stations_tensor, soul,pos_tensor)
        move_deque.append(move)
        print("攻击:",action,"移动：",move)

        
        # action = 0,执行行动wwk
        take_direction(move)
        take_action(action)
        
        # print(time.time() - start_time, " action: ", action_name[action])
        # start_time = time.time()
        
        next_station = thread1.get_buffer()
        next_boss_hp_value = hp.get_boss_hp()
        next_self_hp = hp.get_self_hp()
        next_boss_x = hp.get_hornet_location()
        next_self_x = hp.get_play_location()
        next_pos = [next_self_x/35, next_boss_x/35, next_boss_x/35 - next_self_x/35]

        # get reward
        move_reward, done = Tool.Helper.move_judge(boss_hp_value, next_boss_hp_value,self_hp, next_self_hp,self_x,next_self_x,
                                                   boss_x,next_boss_x,action,move_deque)
        # print(move_reward)
        act_reward, done = Tool.Helper.action_judge(boss_hp_value, next_boss_hp_value,self_hp, next_self_hp,self_x,next_self_x,
                                                   boss_x,next_boss_x,action)
        # print(reward)
        # print( action_name[action], ", ", move_name[d], ", ", reward)
        
        DelayMoveReward.append(move_reward)
        DelayActReward.append(act_reward)
        DelayStation.append(stations)
        DelayActions.append(action)
        DelayDirection.append(move)
        DelayPosition.append(pos)
        DelayStation_next.append(next_station)
        DelayPosition_next.append(next_pos)

        if len(DelayStation) >= DELAY_REWARD:
            # if DelayMoveReward[0] !=0:
            move_rmp_correct.append((DelayStation[0], DelayDirection[0], DelayMoveReward[0], DelayStation_next[0], DelayPosition[0],DelayPosition_next[0], done))

        if len(DelayStation) >= DELAY_REWARD:
            # if DelayMoveReward[0] != 0:
            act_rmp_correct.append((DelayStation[0], DelayActions[0], mean(DelayActReward), DelayStation_next[0], DelayPosition[0],DelayPosition_next[0], done))

        station = next_station
        self_hp = next_self_hp
        boss_hp_value = next_boss_hp_value


        total_act += act_reward
        total_move += move_reward
        # paused = Tool.Helper.pause_game(paused)

        if done == 1:
            Tool.Actions.Nothing()
            break
        elif done == 2:
            PASS_COUNT += 1
            Tool.Actions.Nothing()
            time.sleep(3)
            break
        

    thread1.stop()
    print("训练")
    for i in range(22):
        print("训练中")
        if (len(move_rmp_correct) > MEMORY_WARMUP_SIZE):
            # print("move learning")
            batch_station,batch_actions,batch_reward,batch_next_station,batch_pos,batch_next_pos,batch_done = move_rmp_correct.sample(BATCH_SIZE)
            batch_station = to_tensor(batch_station)
            batch_next_station = to_tensor(batch_next_station)
            batch_actions = torch.tensor(batch_actions, dtype=torch.long).to(device)
            batch_reward = torch.tensor(batch_reward, dtype=torch.float32).to(device)
            batch_pos = torch.tensor(batch_pos, dtype=torch.float32).to(device)
            batch_next_pos = torch.tensor(batch_next_pos, dtype=torch.float32).to(device)

            loss_move = algorithm.move_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_pos,batch_next_pos,batch_done)

        if (len(act_rmp_correct) > MEMORY_WARMUP_SIZE):
            # print("action learning")wwk
            batch_station,batch_actions,batch_reward,batch_next_station,batch_pos,batch_next_pos,batch_done = act_rmp_correct.sample(BATCH_SIZE)
            batch_station = to_tensor(batch_station)
            batch_next_station = to_tensor(batch_next_station)
            batch_actions = torch.tensor(batch_actions, dtype=torch.long).to(device)
            batch_reward = torch.tensor(batch_reward, dtype=torch.float32).to(device)
            batch_pos = torch.tensor(batch_pos, dtype=torch.float32).to(device)
            batch_next_pos = torch.tensor(batch_next_pos, dtype=torch.float32).to(device)

            loss_act = algorithm.act_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_pos,batch_next_pos,batch_done)
        if (len(act_rmp_correct) > MEMORY_WARMUP_SIZE):
            algorithm.shared_learn(loss_move,loss_act)
    print("训练结束")
    # if episode==1 or episode==2 or episode==3:
    #     time.sleep(5.5)
    return total_act,total_move, step, PASS_COUNT, self_hp, greedy


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_remind_hp = 0
    writer = SummaryWriter("runs/hollow_knight_dqn")
    act_rmp_correct = ReplayMemory(MEMORY_SIZE, file_name='./act_memory')   # experience pool
    move_rmp_correct = ReplayMemory(MEMORY_SIZE,file_name='./move_memory')         # experience pool
    act_rmp_correct.load('./act_memory/memory_0.txt')
    move_rmp_correct.load('./move_memory/memory_0.txt')
    # new model, if exit save file, load itk
    model = Model(INPUT_SHAPE, ACTION_DIM)
    model.to(device)
    move_deque = collections.deque(maxlen=6)
    # Hp counter
    hp = Hp_getter()


    model.load_model()
    # model.load_target()
    algorithm = DQN(model, gamma=GAMMA, learning_rate=LEARNING_RATE)
    agent = Agent(ACTION_DIM,algorithm,e_greed=0.95,e_greed_decrement=2e-5)
    # get user input, no need anymore
    # user = User()

    # paused at the begining
    paused=True
    # paused = Tool.Helper.pause_game(paused)

    max_episode = 30000
    # 开始训练
    episode = 0
    PASS_COUNT = 0
    reward_history = []# pass count
    while episode < max_episode:    # 训练max_episode个回合，test部分不计算入episode数量
        # 训练
        episode += 1     
        if episode % 20 == 1:
            algorithm.replace_target()

        total_act, total_move,total_step, PASS_COUNT, remind_hp, greedy = run_episode(hp, algorithm,agent,act_rmp_correct, move_rmp_correct, PASS_COUNT, paused,device,move_deque,episode)
        writer.add_scalar("Act_Reward", total_act, episode)
        writer.add_scalar("Move_Reward", total_move, episode)
        writer.add_scalar("remain_hp", remind_hp, episode)
        writer.add_scalar("steps", total_step, episode)
        print("当前经验池大小:", len(act_rmp_correct),len(move_rmp_correct))
        if episode % 5 == 1:
            model.save_model(episode)
        if episode % 5 == 0:
            move_rmp_correct.save(move_rmp_correct.file_name)
        if episode % 5 == 0:
            act_rmp_correct.save(act_rmp_correct.file_name)
        total_remind_hp += remind_hp
        print("Episode: ", episode,",Act_reward: ",total_act, ",Move_reward: ",total_move,",greedy: ",greedy,", pass_count: " , PASS_COUNT, ", hp:", total_remind_hp / episode)