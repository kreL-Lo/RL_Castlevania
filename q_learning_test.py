import numpy as np
import gym

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT =0.95
EPISODES =25000
SHOW = 500

epsilon =0.5
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = EPISODES // 2

epsilon_decay_val =epsilon/(END_EPSILON_DECAY - START_EPSILON_DECAY)

DISCRETE_OS_SIZE =[20]* len(env.observation_space.high)

print(DISCRETE_OS_SIZE,len(env.observation_space.high))
discrete_os_win_size =(env.observation_space.high- env.observation_space.low)/DISCRETE_OS_SIZE
print(discrete_os_win_size,'here')

q_table = np.random.uniform(low=-2,high =0,size = (DISCRETE_OS_SIZE+[env.action_space.n]))
print(env.observation_space.high, env.observation_space)

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

