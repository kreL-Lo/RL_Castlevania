import numpy as np
from agent import Agent
import retro
from collections import deque
from util import preprocess_frame, stack_frames, get_best_folder

if __name__ == '__main__':

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

    env = retro.make(game='Castlevania-Nes')
    obs = env.reset()

    # moving to fist level.
    action_index = 1  # move forward
    for i in range(850):
        env.step(possible_actions[action_index])
        env.render()

    folder = get_best_folder()

    if folder['path'] is not False:
        path = folder['path']
    else:
        path = False

    agent = Agent(stack_shape, path, action_number)
    agent.epsilon = agent.epsilon_min
    stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)
    state, stacked_frames = stack_frames(stacked_frames, obs, True)
    done = False

    while not done:
        action_index = agent.act(state)
        next_state, reward, done, stats = env.step(possible_actions[action_index])
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        env.render()
        state = next_state


