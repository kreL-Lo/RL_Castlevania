from tensorflow import keras
import retro
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
sys.path.insert(1, '..\DNQ')
import util 
import time
import MultiBinary
model = keras.models.load_model('models/Catlevania-1')

possible_actions = np.array(
    [
    [0,0,0, 0,0,1, 0,0,0],#sit
    [0,0,0, 0,0,0 ,1,0,0],#back_movement
    [0,0,0, 0,0,0 ,0,1,0],#forward
    [0,0,0, 0,0,0 ,0,0,1],#move_forward
    [1,0,0, 0,0,0 ,0,0,0],#jump
    ]
)
print(possible_actions)
nr_actions = len(possible_actions)

def parse_action(nr):
    return possible_actions[nr]




env = retro.make(game='Castlevania-Nes',use_restricted_actions=retro.Actions.ALL, state='Level1')
env.reset()
stacked_frames = util.stacked_frames
stack_frames = util.stack_frames

frame, rew,done,_  = env.step(parse_action(0))
init =False
def get_qs(state,model):
        current_state = util.preprocess_frame(state)
        current_state = current_state.reshape(110,84,1)
        d  = model.predict(np.array(current_state).reshape((-1,*current_state.shape)))
        print("d:",d)
        return d
done = False
obs = env.reset()
state, stacked_frames = stack_frames(stacked_frames, obs, True)

state , rew,done,stats =  env.step(parse_action(0))

state, stacked_frames = stack_frames(stacked_frames, obs, False)

state , rew,done,stats =  env.step(parse_action(0))

state, stacked_frames = stack_frames(stacked_frames, obs, False)
state , rew,done,stats =  env.step(parse_action(0))

state, stacked_frames = stack_frames(stacked_frames, obs, False)



qs = get_qs(state,model)


print('buttons',env.buttons)
while(True):
    env.step([1,0,0, 0,0,0, 0,1,0])
    env.render()
# while not done:
#     # print(env.action_space.sample())
#     env.step([0,0,0 ,0,0,0, 0,0,1])
#     env.render()
#     env.step([0,0,0 ,0,0,0, 1,1,1])
#     env.render()
#     env.step([0,0,0 ,0,0,1, 1,1,1])
#     env.render()
#     env.step([1,0,0 ,0,0,0, 1,1,1])
#     env.render()
#     env.step([0,0,0 ,0,0,0, 1,1,1])
#     env.render()
#     env.step([0,0,0 ,0,0,1, 1,1,1])
#     env.render()
    # print('done')


    # qs = get_qs(state,model)
    # #print(np.argmax(qs), qs)
    # action = np.argmax(qs)
    # #d = model.predict(frame)
    # m = np.zeros(9)
    # m[action]=1
    # print("action:", action)
    # state , rew,done,stats =  env.step(m)
    # env.render
    # env.step(possible_actions[3])
    # state, stacked_frames = stack_frames(stacked_frames, state, False)
    # env.render()


