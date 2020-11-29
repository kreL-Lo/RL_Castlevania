from tensorflow import keras
import retro
import util 
import numpy as np
import matplotlib.pyplot as plt
model = keras.models.load_model('models/CASTLE-12')

possible_actions = np.array(
    [[0,0,0, 0,0,0 ,1,0,0],#back_movement
    [0,0,0, 0,0,0 ,0,1,0],#forward_movement
    [0,0,0, 0,0,0 ,0,0,1],#jump
    [1,0,0, 0,0,0 ,0,0,0],#attack
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


env = retro.make(game='Castlevania-Nes',state='Level1')
env.reset()
move_to_first_level(env)
frame, rew,done,_  = env.step(parse_action(0))
def get_qs(state,model):
        d  = model.predict(np.array(state).reshape(-1,*state.shape))
        #print(d)
        return d[0]
while True:
    
    frame  = util.preprocess_frame(frame)
    
    frame = frame.reshape(110,84,1)
    qs = get_qs(frame,model)
    action = np.argmax(qs)
    #d = model.predict(frame)
    frame , rew,done,_ =  env.step(parse_action(action))
    #print(d)
    env.render()


