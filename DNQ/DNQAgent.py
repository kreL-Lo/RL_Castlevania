import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D,MaxPooling2D, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam 
from collections import deque
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#plt.imshow(mpimg.imread('MyImage.png'))

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1000
MODEL_NAME = "256x2"
MINI_BATCH_SIZE = 64
DISCOUNT =0.99
UPDATE_TARGET_EVERY = 5

class DNQAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0


    def create_model(self):
        model = Sequencial()
        model.add( Conv2D(256,(3,3)))   
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))
        model.add( Conv2D(256,(3,3)))   
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(activation="linear"))
        model.compile(loss="mse",optimizer =Adam(lr=0.001),metrics =['accuracy'])
        return model
    def update_replay_memory(self,transition):
        self.replay_memory.append(transition)
    def get_qs(self,state,step):
        return self.model_predict(np.array(state).reshape(-1,*state.shape)/255.0)[0]
    def train(self,terminal_state,step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory,MINI_BATCH_SIZE)
        
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])/255

        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        Y = []
        for index, (curent_state, action,reward,new_current_state, done) in enumerate(minibatch):
            if not done:
                max_futureq_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT* max_futureq_q
            else :
                new_q = reward
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            X.append(current_state)
            Y.append(current_qs)

        self.model.fit(np.array(X)/255,np.array(y), batch_size=MINI_BATCH_SIZE ,verbose = 0, shuffle = False)

        if terminal_state:
            self.target_update_counter +=1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_model.update_counter =0
        



