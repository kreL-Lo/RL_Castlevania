import retro
import sys
sys.path.insert(1, '..\DNQ')

movie_path = '.\human\Castlevania-Nes\contest\Castlevania-Nes-Level1-0006.bk2'
movie = retro.Movie(movie_path)
movie.step()

env = retro.make(game=movie.get_game(), state=retro.State.NONE, use_restricted_actions=retro.Actions.ALL)
env.initial_state = movie.get_state()
env.reset()
import util 

from Agent import Agent
shape = (110,84,1)
agent = Agent(9,shape,False)

print('stepping movie')
for episode in range(1,100):
    step = 0
    current_state = env.reset()
    current_state = util.preprocess_frame(current_state)
    current_state = current_state.reshape(110,84,1)
    while movie.step():
        keys = []
        for i in range(len(env.buttons)):
            keys.append(movie.get_key(i, 0))
        print("keys:", keys)
        _obs, _rew, _done, _info = env.step(keys)
        # env.render()
        saved_state = env.em.get_state()
        new_state = util.preprocess_frame(_obs)
        new_state = new_state.reshape(110,84,1)
        agent.update_replay_memory((current_state,keys,_rew,new_state,_done))
        step +=1 
        agent.train(step,_done)
    if episode%1==0:
        path = 'models/'+'Catlevania'+"-"+str(episode)
        agent.model.save(path)
    print(episode)


print('stepping environment started at final state of movie')
# env.initial_state = saved_state
env.reset()
while True:
    env.render()
    env.step(env.action_space.sample())