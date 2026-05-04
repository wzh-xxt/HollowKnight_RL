# Define the actions we may need during training
# You can define your actions here

from Tool.SendKey import PressKey, ReleaseKey
from Tool.WindowsAPI import grab_screen
import time
import cv2
import threading

# Hash code for key we may use: https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes?redirectedfrom=MSDN
UP_ARROW = 0x26
DOWN_ARROW = 0x28
LEFT_ARROW = 0x25
RIGHT_ARROW = 0x27

W = 0x57   # 上
A = 0x41   # 左
S = 0x53   # 下
D = 0x44

L = 0x4C
SPACE = 0x20
J = 0x4A
K = 0x4B

L_SHIFT = 0xA0
A = 0x41
C = 0x43
X = 0x58
Z = 0x5A

# move actions
# 0
def Nothing():
    ReleaseKey(A)
    ReleaseKey(D)
    pass

# Move
# 0
def Move_Left():
    PressKey(A)
    time.sleep(0.01)
# 1
def Move_Right():
    PressKey(D)
    time.sleep(0.01)

# 2
def Turn_Left():
    PressKey(A)
    time.sleep(0.01)
    ReleaseKey(A)

# 3
def Turn_Right():
    PressKey(D)
    time.sleep(0.01)
    ReleaseKey(D)

def Jump():
    PressKey(K)
    time.sleep(0.18)
    ReleaseKey(K)

def Quik():
    PressKey(L)
    time.sleep(0.1)
    ReleaseKey(L)


# ----------------------------------------------------------------------

# other actions
# Attackk
# 0

def Attack_left():
    PressKey(A)
    time.sleep(0.01)
    ReleaseKey(A)
    PressKey(J)
    time.sleep(0.1)
    ReleaseKey(J)

def Attack_right():
    PressKey(D)
    time.sleep(0.01)
    ReleaseKey(D)
    PressKey(J)
    time.sleep(0.1)
    ReleaseKey(J)
# 1
def Attack_Down():
    PressKey(S)
    PressKey(J)
    time.sleep(0.1)
    ReleaseKey(J)
    ReleaseKey(S)
    time.sleep(0.01)
# 1
def Attack_Up():
    PressKey(W)
    PressKey(J)
    time.sleep(0.1)
    ReleaseKey(J)
    ReleaseKey(W)
    Nothing()
    time.sleep(0.01)

def Skill_Up():
    PressKey(W)
    PressKey(SPACE)
    time.sleep(0.1)
    ReleaseKey(W)
    ReleaseKey(SPACE)
    Nothing()
    time.sleep(0.15)
# 5
def Skill_Down():
    PressKey(S)
    PressKey(SPACE)
    time.sleep(0.1)
    ReleaseKey(S)
    ReleaseKey(SPACE)
    Nothing()
    time.sleep(0.15)

# Restart function
# it restart a new game
# it is not in actions space
def Look_up():
    PressKey(W)
    time.sleep(0.2)
    ReleaseKey(W)

def restart():
    station_size = (230, 230, 1670, 930)
    print("重开")
    while True:
        print(1)
        station = cv2.resize(cv2.cvtColor(grab_screen(station_size), cv2.COLOR_RGBA2RGB),(1000,500))
        if station[187][300][0] != 0: 
            time.sleep(1)
        else:
            break
    time.sleep(3.5)
    Look_up()
    time.sleep(2.5)
    Look_up()
    time.sleep(1.5)
    while True:
        station = cv2.resize(cv2.cvtColor(grab_screen(station_size), cv2.COLOR_RGBA2RGB),(1000,500))
        if station[187][300][0] == 0:
            # PressKey(DOWN_ARROW)
            # time.sleep(0.1)wwk
            # ReleaseKey(DOWN_ARROW)
            print(3)
            PressKey(K)
            time.sleep(0.2)
            ReleaseKey(K)
            break
        else:
            Look_up()
            time.sleep(0.2)

def restart1():
    station_size = (230, 230, 1670, 930)
    print("重开")
    while True:
        station = cv2.resize(cv2.cvtColor(grab_screen(station_size), cv2.COLOR_RGBA2RGB),(1000,500))
        if station[187][300][0] != 0:
            time.sleep(1)
        else:
            break
    time.sleep(3.5)
    Look_up()
    time.sleep(2.5)
    Look_up()
    time.sleep(2.5)
    while True:
        station = cv2.resize(cv2.cvtColor(grab_screen(station_size), cv2.COLOR_RGBA2RGB),(1000,500))
        if station[187][300][0] == 0:
            PressKey(S)
            time.sleep(0.1)
            ReleaseKey(S)
            print(3)
            PressKey(K)
            time.sleep(0.2)
            ReleaseKey(K)
            break
        else:
            Look_up()
            time.sleep(0.2)


# List for action functions
Actions = [Attack_Down,  Skill_Up,
           Skill_Down, Attack_right,Attack_left]
Directions = [Move_Left, Move_Right, Turn_Left, Turn_Right, Jump, Quik,Nothing]
Total = [Move_Left, Move_Right, Turn_Left, Turn_Right, Jump, Quik,
         Attack_Down,  Skill_Up,Skill_Down, Attack_right,Attack_left]

# Run the actionk
def take_action(action):
    Actions[action]()

def take_direction(direc):
    Directions[direc]()

def take_total(total):
    Total[total]()



class TackAction(threading.Thread):
    def __init__(self, threadID, name, direction, action):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.direction = direction
        self.action = action
        
    def run(self):
        take_direction(self.direction)
        take_action(self.action)
