#%%
import time
import pickle
import shutil
import pandas as pd
from collections import deque
import random
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.models import load_model
import matplotlib.pyplot as plt
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
import gc

from pympler import asizeof
from pympler import muppy, summary

from rocketSim import RocketSim
env = RocketSim()
np.random.seed(0)


class DQN:


    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1
        self.gamma = .95
        self.batch_size = 32
        self.epsilon_min = .01
        self.epsilon_decay = .995
        self.learning_rate = 0.001

        ############################## IMPORTANT #############################
        self.memory = deque(maxlen=500000) # TO CREATE NEW MEMORY
        # with open("Trained_Models\SUCCESS_2023_06_03-13_04\epoch_1000_memory.pickle", "rb") as file:
        #     # Step 3: Load the pickled object using pickle.load()
        #     self.memory = pickle.load(file)
        ######################################################################


        ############################### IMPORTANT #############################
        self.model_path = None # USE THIS TO CREATE A NEW MODEL
        # self.model_path = "Trained_Models\SUCCESS_2023_06_03-13_04\model_episode_1000_2023_06_04-03_53.h5" # USE THIS TO LOAD A MODEL
        ########################################################################



        if self.model_path is None:
            self.model = self.build_model()
        else:
            self.model = load_model(self.model_path)

    def build_model(self):

        model = Sequential()
        model.add(Dense(128, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



def train_dqn(episode):
    global foldername

    #################### IMPORTANT ######################################
    loss = [] # CREATE NEW LOSS
    # with open("Trained_Models\SUCCESS_2023_06_03-13_04\loss_epochs_1000", "rb") as file:
    #     # Step 3: Load the list using pickle.load()
    #     loss = pickle.load(file) # Continue from previous loss
    #######################################################################
    action_space = 3
    state_space = 4
    max_steps = 800
    agent = DQN(action_space, state_space)
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, state_space))
        score = 0

        print("starting episode: {}/{}".format(e, episode))
        print('Score Beginning: {:.2f}'.format(score))
        # create a string with the current date and time for file naming
        timestamp = time.strftime("%Y_%m_%d-%H_%M")
        for i in range(max_steps):
            action = agent.act(state)
            reward, next_state, done = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, state_space))
            agent.remember(state, action, reward, next_state, done)
            del state
            state = next_state
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
            if i % 25 == 0:
                print("step: {}".format(i), "action: {}".format(action))

            # Delete variables or objects that are no longer needed

            # del action
            # del reward
            # del next_state
            # del done

            # Perform garbage collection to free up memory
            # gc.collect()

            # Animation
            # if e == 0:
            #     env.update_animation()

        # Save model halfway through training
        if e % 100 == 0 or e == 1 or e == 50:
            if agent.model_path is None:
                timestamp = time.strftime("%Y_%m_%d-%H_%M")

                # Save the current version of the network
                file_path = f"Trained_Models/{foldername}/model_episode_{e}_{timestamp}.h5"
                agent.model.save(file_path)

                # Save a current version of the loss plot
                window_size = 75
                rolling_avg = pd.Series(loss).rolling(window_size).mean()

                fig1, ax1 = plt.subplots(figsize=(10, 8))
                x = [i for i in range(e)]
                ax1.plot(x[1:], loss[1:], label='Reward')
                ax1.plot(x[window_size:], rolling_avg[window_size:], label='75-episode Rolling Average')
                ax1.set_xlabel('episodes')
                ax1.set_ylabel('reward')
                ax1.legend()
                timestamp = time.strftime("%Y_%m_%d-%H_%M")
                fig_file_path = f"Trained_Models/{foldername}/{timestamp}.png"
                fig1.savefig(fig_file_path)

                # Save a current list of the loss
                with open(f"Trained_Models/{foldername}/loss_epochs_{e}", 'wb') as f:
                    pickle.dump(loss, f)
                # Save a current version of the memory
                with open(f"Trained_Models/{foldername}/epoch_{e}_memory.pickle", 'wb') as f:
                    pickle.dump(agent.memory, f)
                gc.collect()
            else:
                timestamp = time.strftime("%Y_%m_%d-%H_%M")

                # Save the current version of the network
                file_path = f"Trained_Models/{foldername}/prevModelContinuedFrom750_episode_{e}_{timestamp}.h5"
                agent.model.save(file_path)

                # Save a current version of the loss plot
                window_size = 75
                rolling_avg = pd.Series(loss).rolling(window_size).mean()

                fig1, ax1 = plt.subplots(figsize=(10, 8))
                x = [i for i in range(len(loss))]
                ax1.plot(x[1:], loss[1:], label='Reward')
                ax1.plot(x[window_size:], rolling_avg[window_size:], label='75-episode Rolling Average')
                ax1.set_xlabel('episodes')
                ax1.set_ylabel('reward')
                ax1.legend()
                timestamp = time.strftime("%Y_%m_%d-%H_%M")
                fig_file_path = f"Trained_Models/{foldername}/continuedfrom750{timestamp}.png"
                fig1.savefig(fig_file_path)

                # Save a current list of the loss
                with open(f"Trained_Models/{foldername}/loss_epochs_{e}", 'wb') as f:
                    pickle.dump(loss, f)
                # Save a current version of the memory
                with open(f"Trained_Models/{foldername}/epoch_{e}_memory.pickle", 'wb') as f:
                    pickle.dump(agent.memory, f)
                gc.collect()

        # if e % 10 == 0 or e == 1:
        #   K.clear_session()
        #   tf.keras.backend.clear_session()

        loss.append(score)
        print('Length of Remember: {:.2f}'.format(len(agent.memory)))
        print('Score End: {:.2f}'.format(score))
        print('Reward End: {:.2f}'.format(env.reward))

        # Delete variables or objects at the end of each episode
        # del state
        # del agent

        # Perform garbage collection to free up memory
        gc.collect()


    # set the file path including the current timestamp
    file_path = f"Trained_Models/{foldername}/model_episode{episode}_{timestamp}.h5"

    # save the model to the file path
    agent.model.save(file_path)

    # save loss to the folder for future reference
    with open(f"Trained_Models/{foldername}/loss_epochs_{episode}", 'wb') as f:
        pickle.dump(loss, f)

    return loss

if __name__ == '__main__':


    ###################################  IMPORTANT ######################################
    foldername = time.strftime("%Y_%m_%d-%H_%M") # IF NEW MODEL IS TO BE CREATED USE THIS
    # foldername = "ContinuationOf_SUCCESS_2023_06_03-13_04_currentdateis_6_5_2023" # IF CONTINUING A MODEL SPECIFY THE FOLDER NAME
    #####################################################################################



    if not os.path.isdir(f'Trained_Models/{foldername}'):
        os.mkdir(f'Trained_Models/{foldername}')
    # Save current versions of code for reference
    shutil.copy('agentDef.py', f'Trained_Models/{foldername}/agentDef.py')
    shutil.copy('rocketSim.py', f'Trained_Models/{foldername}/rocketSim.py')

    # Begin training of specified episodes

    ep = 10000
    loss = train_dqn(ep)
    #%% ############### Create figure of loss plot #######################
    # plt.figure()

    # window_size = 75
    # rolling_avg = pd.Series(loss).rolling(window_size).mean()

    # fig1, ax1 = plt.subplots(figsize=(10, 8))
    # x = [i for i in range(e)]
    # ax1.plot(x[1:], loss[1:], label='Reward')
    # ax1.plot(x[window_size:], rolling_avg[window_size:], label='75-episode Rolling Average')
    # ax1.set_xlabel('episodes')
    # ax1.set_ylabel('reward')
    # ax1.legend()
    # timestamp = time.strftime("%Y_%m_%d-%H_%M")
    # fig_file_path = f"Trained_Models/{foldername}/{timestamp}.png"
    # fig1.savefig(fig_file_path)

    plt.show()

    print(loss)
# %%
