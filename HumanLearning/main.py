import retro

movie_path = '.\human\Castlevania-Nes\contest\Castlevania-Nes-Level1-0000.bk2'
movie = retro.Movie(movie_path)
movie.step()

env = retro.make(game=movie.get_game(), state=retro.State.NONE, use_restricted_actions=retro.Actions.ALL)
env.initial_state = movie.get_state()
env.reset()

def create_model(nr_actions,shape):
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
    model.add(Dense(activation="elu",units = nr_actions))
    adam = Adam(lr=LR)
    model.compile(loss='categorical_crossentropy',optimizer=adam)
    return model

shape = (110,84,1)
agent = Agent(6,shape)

create_model()

print('stepping movie')
while movie.step():
    keys = []
    for i in range(len(env.buttons)):
        keys.append(movie.get_key(i, 0))
    _obs, _rew, _done, _info = env.step(keys)
    env.render()
    saved_state = env.em.get_state()

print('stepping environment started at final state of movie')
env.initial_state = saved_state
env.reset()
while True:
    env.render()
    env.step(env.action_space.sample())