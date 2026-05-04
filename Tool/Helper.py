from Tool.WindowsAPI import key_check
import time


# check whether a game is end
def is_end(next_self_blood, min_hp, next_boss_blood, boss_blood):
    if next_self_blood ==9 and min_hp <= 3:    
        return True
    elif next_boss_blood - boss_blood > 200:   
        return True
    return False

# get mean score of a reward seq
def mean(d):
    t = 0
    for i in d:
        t += i
    return t / len(d)

# count play hp change, and give reward 
def count_self_reward(next_self_blood, self_hp):
    if next_self_blood - self_hp < 0:
        return next_self_blood - self_hp
    return 0

# count boss hp change, and give reward 
def count_boss_reward(next_boss_blood, boss_blood):
    if next_boss_blood -  boss_blood < 0:
        return int((boss_blood - next_boss_blood)/19)
    return 0

def attack_distance_reward(next_player_x,next_hornet_x,next_boss_blood, boss_blood):
    dist = abs(next_player_x-next_hornet_x)

    if dist < 4.5 and next_boss_blood == boss_blood:
        return -1.5
    if dist < 4.5 and next_boss_blood < boss_blood:
        return 1.6

    return 0


def constant_move_reward(move_deque,self_x_deque, move):
    from collections import Counter
    move_punish = 0
    if len(self_x_deque) >= 6:
        counts = Counter(move_deque)
        most_common_action, most_common_count = counts.most_common(1)[0]

        repeat_rate = most_common_count / len(move_deque)
        if repeat_rate >= 2/3 and move == most_common_action:
            move_punish = -1

    pos_punish = 0
    if len(self_x_deque) >= 3:
        danger_steps = 0
        for x in self_x_deque:
            if x < 15.5 or x > 37.1:
                danger_steps += 1

        if danger_steps >= 3:
            pos_punish = -1

    return pos_punish + move_punish

def move_distance(next_player_x,next_hornet_x):
    if abs(next_player_x - next_hornet_x) < 1.8:
        return -0.4
    elif abs(next_player_x - next_hornet_x) < 4.5:
        return 0.6
    else:
        return -0.2

def facing_reward(self_x,boss_x, move):
    if (self_x - boss_x > 4.2) and move==0:
        return +0.35
    if (self_x - boss_x < 4.2) and (self_x - boss_x >2) and move==2:
        return +0.2
    if (boss_x - self_x > 4.2) and move==1:
        return +0.35
    if (boss_x - self_x < 4.2) and (boss_x - self_x >2) and move==3:
        return +0.2
    if move==4 or move==5:
        return +0.1
    return 0

def total_judge(boss_blood, next_boss_blood, self_blood, next_self_blood,self_x,next_self_x,
                                                           boss_x,next_boss_x,move,move_deque,self_x_deque):
    survival_bonus=0
    if next_self_blood <= 0 and self_blood != 8:
        boss_blood_reward = count_boss_reward(next_boss_blood, boss_blood)
        attack_distance = attack_distance_reward(next_self_x,next_boss_x,next_boss_blood, boss_blood)
        self_blood_reward = count_self_reward(next_self_blood, self_blood)
        move_reward = move_distance(next_self_x,next_boss_x)
        constant_move = constant_move_reward(move_deque,self_x_deque,move)
        facing = facing_reward(self_x, boss_x, move)
        reward = 2 * self_blood_reward + move_reward + constant_move  + survival_bonus + facing + boss_blood_reward + attack_distance
        # print(self_blood_reward,0.5 * boss_blood_reward,move_reward,constant_move)
        done = 1
        return reward, done

    #boss dead
    elif next_boss_blood <= 0 or next_boss_blood > 900:
        boss_blood_reward = count_boss_reward(next_boss_blood, boss_blood)
        attack_distance = attack_distance_reward(next_self_x,next_boss_x,next_boss_blood, boss_blood)
        self_blood_reward = count_self_reward(next_self_blood, self_blood)
        move_reward = move_distance(next_self_x,next_boss_x)
        constant_move = constant_move_reward(move_deque,self_x_deque,move)
        facing = facing_reward(self_x, boss_x, move)
        reward = 2 * self_blood_reward + move_reward + constant_move  + survival_bonus + facing + boss_blood_reward + attack_distance
        # print(self_blood_reward,0.5 * boss_blood_reward,move_reward,constant_move)
        done = 1
        return reward, done
    # playing
    else:
        boss_blood_reward = count_boss_reward(next_boss_blood, boss_blood)
        attack_distance = attack_distance_reward(next_self_x,next_boss_x,next_boss_blood, boss_blood)
        self_blood_reward = count_self_reward(next_self_blood, self_blood)
        move_reward = move_distance(next_self_x,next_boss_x)
        constant_move = constant_move_reward(move_deque,self_x_deque,move)
        facing = facing_reward(self_x, boss_x, move)
        reward = 2 * self_blood_reward + move_reward + constant_move  + survival_bonus + facing + boss_blood_reward + attack_distance
        # print(self_blood_reward,0.5 * boss_blood_reward,move_reward,constant_move)
        done = 0
        return reward, done


# JUDGEMENT FUNCTION, write yourself
def action_judge(boss_blood, next_boss_blood, self_blood, next_self_blood,self_x,next_self_x,
                                                   boss_x,next_boss_x,action):
    # Player dead
    if next_self_blood <= 0 and self_blood != 8:
        boss_blood_reward = count_boss_reward(next_boss_blood, boss_blood)
        attack_distance = attack_distance_reward(next_self_x,next_boss_x,next_boss_blood, boss_blood)
        reward = boss_blood_reward + attack_distance
        # print(0.5*self_blood_reward,boss_blood_reward,attack_distance)
        # reward = boss_blood_reward
        if action == 2 and (next_boss_blood - boss_blood) < 0:
            reward *= 3
        done = 1
        return reward, done

    #boss dead
    elif next_boss_blood <= 0 or next_boss_blood > 900:
        boss_blood_reward = count_boss_reward(next_boss_blood, boss_blood)
        attack_distance = attack_distance_reward(next_self_x,next_boss_x,next_boss_blood, boss_blood)
        reward = boss_blood_reward + attack_distance
        # print(0.5*self_blood_reward,boss_blood_reward,attack_distance)
        # reward = boss_blood_reward
        if action == 2 and (next_boss_blood - boss_blood) < 0:
            reward *= 3
        done = 1
        return reward, done
    # playing
    else:
        boss_blood_reward = count_boss_reward(next_boss_blood, boss_blood)
        attack_distance = attack_distance_reward(next_self_x,next_boss_x,next_boss_blood, boss_blood)
        reward = boss_blood_reward + attack_distance
        # print(0.5*self_blood_reward,boss_blood_reward,attack_distance)
        # reward = boss_blood_reward
        if action == 2 and (next_boss_blood - boss_blood) < 0 :
            reward *= 3
        done = 0
        return reward, done


def move_judge(boss_blood, next_boss_blood, self_blood, next_self_blood,self_x,next_self_x,
                                                           boss_x,next_boss_x,move,move_deque,self_x_deque):
    # Player dead
    survival_bonus=0
    if next_self_blood <= 0 and self_blood != 8:
        self_blood_reward = count_self_reward(next_self_blood, self_blood)
        move_reward = move_distance(next_self_x,next_boss_x)
        constant_move = constant_move_reward(move_deque,self_x_deque,move)
        facing = facing_reward(self_x, boss_x, move)
        reward = 2 * self_blood_reward + move_reward + constant_move  + survival_bonus + facing
        # print(self_blood_reward,0.5 * boss_blood_reward,move_reward,constant_move)
        done = 1
        return reward, done

    #boss dead
    elif next_boss_blood <= 0 or next_boss_blood > 900:
        self_blood_reward = count_self_reward(next_self_blood, self_blood)
        move_reward = move_distance(next_self_x,next_boss_x)
        constant_move = constant_move_reward(move_deque,self_x_deque,move)
        facing = facing_reward(self_x, boss_x, move)
        reward = 2 * self_blood_reward + move_reward + constant_move  + survival_bonus + facing
        # print(self_blood_reward,0.5 * boss_blood_reward,move_reward,constant_move)
        done = 1
        return reward, done
    # playing
    else:
        self_blood_reward = count_self_reward(next_self_blood, self_blood)
        move_reward = move_distance(next_self_x,next_boss_x)
        constant_move = constant_move_reward(move_deque,self_x_deque,move)
        facing = facing_reward(self_x, boss_x, move)
        reward = 2 * self_blood_reward + move_reward + constant_move  + survival_bonus + facing
        # print(self_blood_reward,0.5 * boss_blood_reward,move_reward,constant_move)
        done = 0
        return reward, done

# Paused training
def pause_game(paused):
    op, d = key_check()
    if 'T' in op:
        if paused:
            paused = False
            print('start game')
            time.sleep(1)
        else:
            paused = True
            print('pause game')
            time.sleep(1)
    if paused:
        print('paused')
        while True:
            op, d = key_check()
            # pauses game and can get annoying.
            if 'T' in op:
                if paused:
                    paused = False
                    print('start game')
                    time.sleep(1)
                    break
                else:
                    paused = True
                    time.sleep(1)
    return paused