import retro
import numpy as np
LEARNING_RATE = 0.1
DISCOUNT =0.95
EPISODES =25000
SHOW = 2000

epsilon =0.5
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = EPISODES // 2

epsilon_decay_val =epsilon/(END_EPSILON_DECAY - START_EPSILON_DECAY)


#DISCRETE_OS_SIZE =[255]* len(env.observation_space.high)

#discrete_os_win_size =(env.observation_space.high- env.observation_space.low)/DISCRETE_OS_SIZE

#q_table = np.random.uniform(low=-2,high =0,size = (env.observation_space.n+[env.action_space.n]))
#print(q_table)
print(retro.data)
env = retro.make(game='Castlevania-Nes',state='Level1')
obs = env.reset()
ff = env.action_space.sample()
rand = [1 ,1 ,0 ,0 ,1, 1 ,0 ,1 ,1]
d =0
while True:
    obs, rew, done, info = env.step(rand)
    if rew >d :
        d = rew
    env.render()
    if done:
        obs  = env.reset()
env.close()
