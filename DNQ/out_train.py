from tensorflow import keras
import retro
import util 
import numpy as np
import matplotlib.pyplot as plt
import random
model = keras.models.load_model('models/SPACE-6')

possible_actions = np.array(
    [[0,0,0, 0,0,0 ,1,0,0],#back_movement
    [0,0,0, 0,0,0 ,0,1,0]#forward_movement
    ,#[0,0,0, 0,0,0 ,0,0,1],#jump
    [1,0,0, 0,0,0 ,0,0,0]#attack
    ]
)
print(possible_actions)
nr_actions = len(possible_actions)

def parse_action(nr):
    return possible_actions[nr]

def move_to_first_level(env):
    act = 1 # move forward
    for i in range(0,850):
        env.step(parse_action(act))
        env.render()


env = retro.make(game='SpaceInvaders-Atari2600')
env.reset()
#move_to_first_level(env)
stacked_frames = util.stacked_frames
stack_frames = util.stack_frames

frame, rew,done,_  = env.step(parse_action(0))
init =False
def get_qs(state,model):
        d  = model.predict(np.array(state).reshape((-1,*state.shape)))
        #print(d)
        return d[0]
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


while not done:
    

    qs = get_qs(state,model)
    #print(np.argmax(qs), qs)
    action = np.argmax(qs)
    #d = model.predict(frame)
    state , rew,done,stats =  env.step(parse_action(action))
    
    state, stacked_frames = stack_frames(stacked_frames, state, False)
    env.render()


