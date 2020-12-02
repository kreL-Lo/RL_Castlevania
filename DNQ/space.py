#TODOS : 
# 1.test functions to save/load prediction model
# 2.functionalities for statitics and adjusting parameters for the models, display using tenser board
# 3.OPTIMIZATION
# 4.problems in making the AI in going to different levels (maybe manual made functions to do so, for each level) (done first stage)
# aici vin alte probleme ca pot exista power-uri, boost-uri la fiecare nivel, daca e manual, at el nu o sa ia 
# alta solutie mai grea aici este a face noi enviromentu, la fiecare nivel sa anunte ca so terminat nivelul , si sa dam reward pentru power-uri...
# apoi pentru fiecare nivel va fi antrenat un agent ca sa-l rezolve
# sau este posibil sa termine cu un singur agent , idk that
import shutil
import json
import numpy as np           
import retro                 
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import util 
import time
import math
import tensorflow as tf
env = retro.make(game='SpaceInvaders-Atari2600')

from Agent import Agent
from Agent import MINIBATCH_SIZE
import os

if os.path.exists('logs'):
    shutil.rmtree('logs')


if os.path.exists('models'):
    shutil.rmtree('models')

if os.path.exists('runLog.txt'):
    os.remove('runLog.txt')
    
EPSILON = 1
EPSILON_DECAY = 0.98
MIN_EPSILON = 0.001
SHOW_PREVIEW = 50
EPISODES = 20_000
MAXSTEPTS = 10_000


possible_actions = np.array(
    [[0,0,0, 0,0,0 ,1,0,0],#back_movement :0
    [0,0,0, 0,0,0 ,0,1,0],#forward_movement :1
    #[0,0,0, 0,0,0 ,0,0,1],#jump :2 
    [1,0,0, 0,0,0 ,0,0,0]#attack :3
    ]
)
nr_actions = len(possible_actions)

b = env.reset()
get_frame = util.preprocess_frame(b)

print(util.stack_frames,'sfsdfasdf')
shape = (110,84,4)
agent = Agent(nr_actions,shape)



def parse_action(nr):
    return possible_actions[nr]
bonus = 0

def move_to_first_level(env):
    act = 1 # move forward
    for i in range(0,850):
        env.step(parse_action(act))
        env.render()

stacked_frames = util.stacked_frames
stack_frames = util.stack_frames
for i in range(MINIBATCH_SIZE):
    # If it's the first step
    if i == 0:
        state = env.reset()
        
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
    # Get the next_state, the rewards, done by taking a random action
    choice = random.randint(1,len(possible_actions))-1
    action = possible_actions[choice]
    next_state, reward, done, _ = env.step(action)
    
    #env.render()
    
    # Stack the frames
    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
    
    
    # If the episode is finished (we're dead 3x)
    if done:
        # We finished the episode
        next_state = np.zeros(state.shape)
        
        # Add experience to memory
        agent.update_replay_memory((state, action, reward, next_state, done))
        
        # Start a new episode
        state = env.reset()
        
        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
    else:
        # Add experience to memory
        agent.update_replay_memory((state, action, reward, next_state, done))
        
        # Our new state is now the next_state
        state = next_state

episode = 1
while episode<=EPISODES:
    f = open("runLog.txt","a")
    episode_reward = 0
    step = 1
    current_state = env.reset()
    
    current_state, stacked_frames = stack_frames(stacked_frames, current_state, True)
    start = time.time()
    done = False
    #move_to_first_level(env)
    stats =''
    hp = 64
    recorder=[0,0,0,0]
    print("Begin episode: "+str(episode))
    qs=0
    
    while step<MAXSTEPTS:
        qs1 = agent.get_qs(current_state)
        #print(qs1)
        if np.random.random()> EPSILON:
            #action = np.argmax(agent.get_qs(current_state))
            qs = agent.get_qs(current_state)
            action = np.argmax(qs)
            #print(qs,action)
            #print(action)
        else :
            action = np.random.randint(0,nr_actions)
        new_state , rew, done,stats  = env.step(parse_action(action))

        step +=1
        if step ==MAXSTEPTS:
            done = True
        if done :
            new_state = np.zeros((110,84), dtype=np.int)
            new_state, stacked_frames =stack_frames(stacked_frames,new_state,False)
            step = MAXSTEPTS
            agent.update_replay_memory((current_state, action, reward, new_state, done))
        else : 
            new_state, stacked_frames =stack_frames(stacked_frames,new_state,False)
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            current_state = new_state

        recorder[action]+=1
        episode_reward += rew
        agent.train(done,step)
        '''
        except Exception e:
            agent.model.save(saved_model)
            agent.target_model.save(saved_target_model)
            f1 = open('execution_save.txt','w')
            f1.write(str(episode))
            f1.close() '''


        sum1 = 0
        for x in recorder:
            sum1+=x
        d = [round(recorder[0]/sum1,2),round(recorder[1]/sum1,2),round(recorder[2]/sum1,2)]
        if step %1000==1:
            print(stats,rew,action,qs,episode,d,EPSILON,step,qs1,np.argmax(qs1))
        qs =0 

    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY
        EPSILON = max(MIN_EPSILON, EPSILON)
    
    if episode%2==0:
        path = 'models/'+'SPACE'+"-"+str(episode)
        agent.model.save(path)
        
    
    end = time.time()
    des1 =round( end - start,2)
    stringul = str(episode) +" , " +str(des1)+" , " + str(episode_reward) + " , "+ str(stats) +" , "+ str(d ) +"\n"
    f.write(stringul)
    tf.keras.backend.clear_session()
    print(episode,episode_reward,stats,des1)
    f.close()
    episode+=1