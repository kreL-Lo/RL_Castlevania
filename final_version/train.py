from agent import Agent
from util import preprocess_frame, stack_frames, get_best_folder
import retro
import tensorflow as tf
import random
import numpy as np
import time
from collections import deque

if __name__ == '__main__':
    # variables to be used
    episodes = 20000
    max_steps = 10000
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

    # initializiation of the agent.
    env = retro.make(game='Castlevania-Nes')
    frame = env.reset()
    processed_frame = preprocess_frame(frame)
    folder = get_best_folder()

    if folder['path'] is not False:
        path = folder['path']
        current_episode = int(folder['episode'])
    else:
        path = False
        current_episode = 1

    agent = Agent(stack_shape, path, action_number)

    if path is not False:
        for i in range(current_episode):
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

        agent.epsilon = max(agent.epsilon_min, agent.epsilon)
        current_episode += 1

    # performing random actions in order to fill the replay memory with minibatch_size states
    stacked_frames = deque([np.zeros((110, 84), dtype=np.int)
                            for i in range(stack_size)], maxlen=stack_size)

    for i in range(agent.minibatch_size):
        if i == 0:
            state = env.reset()
            stafte, stacked_frames = stack_frames(stacked_frames, state, True)

        choice = np.random.randint(0, action_number)
        action = possible_actions[choice]
        next_state, reward, done, _ = env.step(action)
        next_state, stacked_frames = stack_frames(
            stacked_frames, next_state, False)

        if done:
            next_state = env.reset()
            agent.remember(state, choice, reward, next_state, done)
            state = next_state
            state, stacked_frames = stack_frames(stacked_frames, state, True)
        else:
            agent.remember(state, choice, reward, next_state, done)
            state = next_state

    # actually training the agent.
    while current_episode < episodes:
        # episode variables
        score = 0
        step = 1
        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        start_time = time.time()
        done = False
        hp = 64
        lives = 4
        # moving to first level
        action_index = 1  # move forward
        for i in range(850):
            env.step(possible_actions[action_index])
            env.render()

        while step < max_steps and done is False:
            action_index = agent.act(state)
            next_state, reward, done, stats = env.step(
                possible_actions[action_index])
            step += 1
            env.render()

            # adjusting the reward if the player moves right
            if action_index == 1:
                reward += 1

            if stats['health'] < hp:
                reward -= 5
                hp = stats['health']

            # adjusting the reward if the player has just lost a life
            if lives > stats['lives']:
                lives = stats['lives']
                reward -= 10

            if step == max_steps:
                done = True

            if done:
                next_state = env.reset()

            next_state, stacked_frames = stack_frames(
                stacked_frames, next_state, False)
            agent.remember(state, action_index, reward, next_state, done)
            state = next_state

            score += reward
            agent.train(done)

            if step % 1000 == 0:
                print("Stats: %s    Last action: %s    Episode: %s    Epsilon: %s     Score: %s" % (
                    str(stats), str(action_index), str(current_episode), str(agent.epsilon), str(score)))

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            agent.epsilon = max(agent.epsilon, agent.epsilon_min)

        if current_episode % 2 == 0:
            path = 'models/CASTLE-' + str(current_episode)
            agent.model.save(path)

        end_time = time.time()

        log = "End of episode %s    Epsilon: %s     Score: %s     Time: %s seconds" % (
            str(current_episode), str(agent.epsilon), str(score), str(end_time - start_time))

        with open('runlog.txt', 'a') as f:
            f.write(log + '\n')
            f.close()

        tf.keras.backend.clear_session()
        current_episode += 1
