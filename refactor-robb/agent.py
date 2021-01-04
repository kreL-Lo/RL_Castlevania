import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, MaxPooling2D, Conv2D, Dropout, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random


class Agent:
    def __init__(self, state_shape, save_path, number_of_actions, model_name="CASTLE", learning_rate=.00002, epsilon=1,
                 epsilon_min=.001, epsilon_decay = .98,
                 tau=.01, replay_memory_size=10000, minibatch_size=64, gamma=.95, update_target_interval=5):
        self.learning_rate = learning_rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.step = 1
        self.target_model_counter = 0
        self.update_target_interval = update_target_interval
        self.state_shape = state_shape
        self.model_name = model_name
        self.action_number = number_of_actions

        log_dir = f"logs/{self.model_name}-{int(self.step)}"
        self.tensorboard = TensorBoard(log_dir=log_dir)

        if save_path is False:
            self.model = self.create_model()
            self.target_model = self.create_model()
        else:
            model = tf.keras.models.load_model(save_path)
            self.model = tf.keras.models.clone_model(model)
            adam = Adam(lr=self.learning_rate)
            self.model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])
            self.target_model = tf.keras.models.clone_model(model)
            self.target_model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=[8, 8], strides=[4, 4], input_shape=self.state_shape, padding='valid'))
        model.add(Activation("relu"))
        model.add(Conv2D(64, kernel_size=[4, 4], strides=[2, 2], padding='valid'))
        model.add(Activation("relu"))
        model.add(Conv2D(64, kernel_size=[3, 3], strides=[2, 2], padding='valid'))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(units=512))
        model.add(Activation('relu'))
        model.add(Dense(units=self.action_number))
        model.add(Activation("relu"))
        adam = Adam(lr=self.learning_rate)
        model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def train(self, is_terminal_step):
        if len(self.replay_memory) < self.minibatch_size:
            return

        minibatch = random.sample(self.replay_memory, self.minibatch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        targets = rewards + self.gamma * (np.amax(self.target_model.predict_on_batch(next_states), axis=1)) * (
                    1 - dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.minibatch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)

        if is_terminal_step:
            self.step += 1
            self.target_model_counter += 1

        if self.target_model_counter > self.update_target_interval:
            model_weights = self.model.get_weights()
            target_model_weights = self.target_model.get_weights()
            for i in range(len(model_weights)):
                target_model_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_model_weights[i]
            self.target_model.set_weights(target_model_weights)

            self.target_model_counter = 0

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_number)
        else:
            prediction = self.model.predict(np.array(state).reshape((1, *self.state_shape)))
            action = np.argmax(prediction[0])
            return action
