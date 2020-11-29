import retro
import numpy as np
import json

LEARNING_RATE = 0.1
DISCOUNT =0.95
EPISODES =25000
SHOW = 2000

env = retro.make(game='Castlevania-Nes',state='Level1')

eta =0.628
gma =0.9
epis =5000

DISCRETE_OS_SIZE =np.ones_like(env.observation_space.high)

discrete_os_win_size =(env.observation_space.high- env.observation_space.low)/DISCRETE_OS_SIZE


#q_table = np.random.uniform(low=-2,high =0,size = (DISCRETE_OS_SIZE+[env.action_space.n]))
#print(env.observation_space.shape[0])
q_table = np.zeros([env.observation_space.shape[0],env.action_space.n])


LEARNING_RATE = 0.1
DISCOUNT =0.95
EPISODES =25000
SHOW = 500

epsilon =0.5
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = EPISODES // 2
def get_discrete_state(state):
    discrete_state = (state-env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))
 
for episode in range(EPISODES):
    if episode %SHOW == 0:
        render =True    
    else :
        render = False
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        if np.random.random() > epsilon:
            print(discrete_state)
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0,env.action_space.n)
        action = np.argmax(q_table[discrete_state])
        new_state , reward ,done, _ = env.step(action)
        if render:
            env.render()

        new_discrete_state = get_discrete_state(new_state)
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1-LEARNING_RATE) * current_q+ LEARNING_RATE *(reward +DISCOUNT *max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >=env.goal_position:
            print(f"Made id {episode}")
            q_table[discrete_state+(action,)] =0
        discrete_state  = new_discrete_state

        if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAY:
            epsilon -=epsilon_decay_val
        
env.close()
