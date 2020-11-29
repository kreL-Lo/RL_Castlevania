

#ADDJUSTABLE VARIABLES
MAX_MEM_SIZE =5000 # MEMORIA DEQUE-ULUI
MINIBATCH_SIZE =120# NR DE BATCH-URI CARE LE IA LA FIT 
MIN_REPLAY_MEMORY_SIZE =  1000  # CAND INCEPE SA IA LA FIT
DISCOUNT = 0.90 # PT Q LEARNING 
UPDATE_TARGET_EVERY = 5 # CAND UPDATEAZA AL DOILEA MODEL 
LR = 0.001

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dense,MaxPooling2D,Conv2D, Dropout,Flatten
from collections import deque
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import time 

class Agent:
    def __init__(self,nr_actions,shape): 
        self.model = self.create_model_1(nr_actions,shape)
        self.replay_memory = deque(maxlen =MAX_MEM_SIZE)
        self.target_update_counter = 0 
        self.target_model = self.create_model_1(nr_actions,shape)

    def create_model(self,nr_actions,shape):
        model = Sequential()
        model.add( Conv2D(32,kernel_size=8,strides=8,input_shape = shape) )
        model.add(Activation("relu"))
        
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))
        
        model.add( Conv2D(32,kernel_size=4,strides=4)   )
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(activation="sigmoid",units = nr_actions))
        adam = Adam(lr=LR)
        model.compile(loss='categorical_crossentropy',optimizer=adam)
        return model
    def create_model_1(self,nr_actions,shape):
        model = Sequential()
        model.add( Conv2D(32,kernel_size=8,strides=4,input_shape = shape,activation="relu") )
        model.add( Conv2D(64,kernel_size=4,strides=4,activation="relu")   )
        model.add( Conv2D(64,kernel_size=3,strides=3,activation="relu")   )
        model.add(Flatten())
        model.add(Dense(units=512))
        model.add(Activation('relu'))
        model.add(Dense(activation="sigmoid",units = nr_actions))
        adam = Adam(lr=LR)
        model.compile(loss="categorical_crossentropy",optimizer =adam,metrics =['accuracy'])
        return model
    def update_replay_memory(self, frame):
        self.replay_memory.append(frame)

    def train(self,terminal_step,step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return 
        
        minibatch = np.array(random.sample(self.replay_memory,MINIBATCH_SIZE))
        cur_states= np.array(minibatch[:,0:1].tolist())[:,0]
        current_qs_list = self.model.predict(cur_states)
        new_current_states = np.array(minibatch[:,3:4].tolist())[:,0]
        future_qs_list = self.target_model.predict(new_current_states)
        X=[]
        Y=[]
        
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            

            current_qs = current_qs_list[index]
            current_qs[action] = new_q
             
            X.append(current_state)
            Y.append(current_qs)
        
        self.model.fit(np.array(X), np.array(Y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        if terminal_step:
            self.target_update_counter +=1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self,state):
        d  = self.model.predict(np.array(state).reshape(-1,*state.shape))
        #print(d)
        return d[0]

