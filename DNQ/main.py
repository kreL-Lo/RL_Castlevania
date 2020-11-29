#TODOS : 
# 1.test functions to save/load prediction model
# 2.functionalities for statitics and adjusting parameters for the models, display using tenser board
# 3.OPTIMIZATION
# 4.problems in making the AI in going to different levels (maybe manual made functions to do so, for each level) (done first stage)
# aici vin alte probleme ca pot exista power-uri, boost-uri la fiecare nivel, daca e manual, at el nu o sa ia 
# alta solutie mai grea aici este a face noi enviromentu, la fiecare nivel sa anunte ca so terminat nivelul , si sa dam reward pentru power-uri...
# apoi pentru fiecare nivel va fi antrenat un agent ca sa-l rezolve
# sau este posibil sa termine cu un singur agent , idk that
import json
import numpy as np           
import retro                 
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import util 


env = retro.make(game='Castlevania-Nes',state='Level1')

from Agent import Agent


EPSILON = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001
SHOW_PREVIEW = 50
EPISODES = 20_000

possible_actions = np.array(
    [[0,0,0, 0,0,0 ,1,0,0],#back_movement
    [0,0,0, 0,0,0 ,0,1,0],#forward_movement
    [0,0,0, 0,0,0 ,0,0,1],#jump
    [1,0,0, 0,0,0 ,0,0,0],#attack
    ]
)
nr_actions = len(possible_actions)

b = env.reset()
get_frame = util.preprocess_frame(b)

print(get_frame.shape)
shape = (110,84,1)
agent = Agent(nr_actions,shape)



def parse_action(nr):
    return possible_actions[nr]
bonus = 0

def move_to_first_level(env):
    act = 1 # move forward
    for i in range(0,850):
        env.step(parse_action(act))
        env.render()




for episode in range(1,EPISODES):
    episode_reward = 0
    step = 1
    current_state = env.reset()
    current_state = util.preprocess_frame(current_state)
    current_state = current_state.reshape(110,84,1)
    done = False
    move_to_first_level(env)
    stats =''
    recorder=[0,0,0,0]
    print(episode)
    while not done:
        if np.random.random()> EPSILON:
            action = np.argmax(agent.get_qs(current_state))
            print('gotten')
        else :
            action = np.random.randint(0,nr_actions)
        new_state , rew, done,stats  = env.step(parse_action(action))
        if int(stats['health']) <60:
            rew -=100
            done = True
        if action ==0:
            rew+=50
        if action ==1 :
            rew-=25

        recorder[action]+=1


            
        new_state = util.preprocess_frame(new_state)
        #plt.imshow(new_state)
        #plt.show() 
        new_state = new_state.reshape(110,84,1)
        episode_reward += rew
        
        #if episode % SHOW_PREVIEW==2:
        #    env.render()
        agent.update_replay_memory((current_state,action,rew,new_state,done))
        current_state = new_state
        step +=1
        #print(stats)
        
        agent.train(step,done)
        #env.render()
        
        #env.render()
        sum1 = 0
        for x in recorder:
            sum1+=x
        d = [round(recorder[0]/sum1,2),round(recorder[1]/sum1,2),round(recorder[2]/sum1,2),round(recorder[3]/sum1,2)]
        print(action,stats,episode,EPSILON,d)
        #env.render()
    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY
        EPSILON = max(MIN_EPSILON, EPSILON)
    
    if episode%2==0:
        path = 'models/'+'CASTLE'+"-"+str(episode)
        agent.target_model.save(path)
    print(episode,episode_reward,stats)
