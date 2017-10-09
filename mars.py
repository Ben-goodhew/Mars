##import
import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter


###veribals
#input
mars_env='CartPole-v0'
games=10000

#change
goal_steps = 500
i_score=50

#no change
env = gym.make(mars_env)
env.reset()
LR = 1e-3


###functions
#data
def mars_data():
    training_data = []
    scores = []
    accepnd_scores = []
    for mars in range(games):
        score = 0
        game_memory = []
        prev_observation = []
        for mars in range(goal_steps):

            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)
            

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward
            if done:
                break

        if score >= i_score:
            accepnd_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print('Average accepted score:', mean(accepnd_scores))
    print('Median accepted score:', median(accepnd_scores))
    print(Counter(accepnd_scores))

    return training_data

#nural network
def mars_neural_network(input_size):
    network = input_data(shape = [None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

#training
def mars_traning(training, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = mars_neural_network(input_size = len(X[0]))

    model.fit({'input':X}, {'targets':y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openaistuff')

    return model


###start
training_data = mars_data()
model = mars_traning(training_data)

scores = []
choices = []

for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for mars in range(goal_steps):
        env.render()
        if len (prev_obs) == 0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs),1))[0])
        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break
    scores.append(score)

print('Average Scores', sum(scores)/len(scores))
print('Choice 1: {}, Choice 2: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))
