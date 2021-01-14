import retro
from movieagent import Agent
from util import preprocess_frame, stack_frames, get_best_folder
import tensorflow as tf
import time
from collections import deque
import numpy as np
from os import listdir
from os.path import isfile, join


def get_movie_array(basePath):
    onlyfiles = [f for f in listdir(basePath) if isfile(join(basePath, f))]
    return onlyfiles


movie_path = '..\HumanLearning\human\Castlevania-Nes\contest'
movies = get_movie_array(movie_path)

possible_actions = np.array(
    [[0, 0, 0, 0, 0, 0, 1, 0, 0],  # back_movement :0
     [0, 0, 0, 0, 0, 0, 0, 1, 0],  # forward_movement :1
     [0, 0, 0, 0, 0, 0, 0, 0, 1],  # jump :2
     [1, 0, 0, 0, 0, 0, 0, 0, 0]  # attack :3
     ]
)
action_number = len(possible_actions)
stack_size = 4  # We stack 4 frames
stack_shape = (110, 84, stack_size)
path = False
agent = Agent(stack_shape, path, action_number)

current_epoch = 1
epochs = 100

while current_epoch <= epochs:
    for movie in movies:
        print("Learning movie: " + movie)
        movie = retro.Movie(movie_path + '\\' + movie)
        movie.step()

        env = retro.make(game=movie.get_game(), state=retro.State.NONE,
                         use_restricted_actions=retro.Actions.ALL)
        env.initial_state = movie.get_state()
        frame = env.reset()

        stacked_frames = deque([np.zeros((110, 84), dtype=np.int)
                                for i in range(stack_size)], maxlen=stack_size)

        state, stacked_frames = stack_frames(stacked_frames, frame, True)

        while movie.step():
            keys = []
            for i in range(len(env.buttons)):
                keys.append(movie.get_key(i, 0))
            print(keys)
            _obs, _rew, _done, _info = env.step(keys)

            next_state, stacked_frames = stack_frames(stacked_frames, _obs, False)
            # agent.remember(state, )
            env.render()
            saved_state = env.em.get_state()





print('stepping movie')
while movie.step():
    keys = []
    for i in range(len(env.buttons)):
        keys.append(movie.get_key(i, 0))

    print("KEYS: ", str(keys))
    _obs, _rew, _done, _info = env.step(keys)
    env.render()
    saved_state = env.em.get_state()

print('stepping environment started at final state of movie')
env.initial_state = saved_state
env.reset()

while True:
    env.render()
    env.step(env.action_space.sample())
