import tensorflow as tf
import random
import gym
import time
import os
import random
import numpy as np
import collections
import os
import gc
import pickle
import zlib
import shutil

tf.keras.backend.clear_session()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[len(gpus)-1], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

env = gym.make('StarGunnerNoFrameskip-v4')
env = gym.wrappers.AtariPreprocessing(env)
checkpointFile = './checkpoints/checkpoint.h5'
checkpointReward = './checkpoints/checkpointReward.npy'


def dqn_model(env):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(8, 8), activation='relu', input_shape=(
            84, 84, 4), strides=(4, 4), data_format="channels_last"),  # should be 84,84,4 #strid 4
        tf.keras.layers.Conv2D(
            64, kernel_size=(4, 4), activation='relu', strides=(2, 2)),  # stride 2
        tf.keras.layers.Conv2D(
            64, kernel_size=(3, 3), activation='relu', strides=(1, 1)),  # stride 1
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(env.action_space.n, activation='linear') 
    ])


def dqn(M, env, startStep=0, stopEpsilon=0.1, stepsToReachStopEpsilon=1000000, memorySize=1000000, gamma=0.99, learningRate=0.00025, minibatchSize=32, cSteps=10000, render=False):

    # lossOverTime = []
    totalRewardOverTime = []

    qModel = dqn_model(env)  # Q
    qHatModel = dqn_model(env)  # Q hat

    if os.path.exists(checkpointReward) and os.path.isfile(checkpointReward):
        print("Loaded total reward over time")
        totalRewardOverTime = np.load(checkpointReward).tolist()

    if os.path.exists(checkpointFile) and os.path.isfile(checkpointFile):
        print("Loaded model")
        del qModel
        qModel = tf.keras.models.load_model(checkpointFile)
    else:
        # , clipvalue=1.0
        qModel.compile(
            loss='mse', optimizer=tf.keras.optimizers.RMSprop(lr=learningRate, rho=0.95, clipvalue=1.0), metrics=['accuracy'])

    qHatModel.set_weights(qModel.get_weights())

    qHatModel.compile(
        loss='mse', optimizer=tf.keras.optimizers.RMSprop(lr=learningRate, rho=0.95, clipvalue=1.0), metrics=['accuracy'])

    replayMemory = collections.deque()

    startEpsilon = max(1 - (1-stopEpsilon) / stepsToReachStopEpsilon * startStep, stopEpsilon)
    epsilon = startEpsilon
    steps = startStep

    shouldReset = True
    fourFrames = collections.deque()

    lives = None

    avgLoss = 0

    while len(replayMemory) < 50000:

        if shouldReset:
            lastFrame = env.reset()
            lastFrame = lastFrame * (1.0/255)
            numNoOp = 0
            fourFrames = collections.deque()
            fourFrames.append(lastFrame)
            fourFrames.append(lastFrame)
            fourFrames.append(lastFrame)
            fourFrames.append(lastFrame)

        if render:
            env.render()

        lastFrames = np.array([np.asarray(fourFrames, dtype=np.float32)])

        nextAction = env.action_space.sample()
        if 1 - epsilon >= random.uniform(0, 1):
            # pick action action
            valueVector = qModel.predict(tf.transpose(
                tf.convert_to_tensor(lastFrames, dtype=tf.float32), perm=[0, 2, 3, 1]), steps=1)
            nextAction = np.argmax(valueVector[0])

        frame, reward, done, envInfo = env.step(nextAction)
        
        #if lives != None and envInfo['ale.lives'] < lives:
        #    done = True
        #elif lives == None:
        #    lives = envInfo['ale.lives']

        frame = frame * (1.0/255)
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        else:
            reward = 0
        fourFrames.popleft()
        fourFrames.append(frame)
        nextFrames = np.array([np.asarray(fourFrames, dtype=np.float32)])

        if len(replayMemory) == memorySize:
            entry = replayMemory.popleft()
            del entry

        replayMemory.append(
            (lastFrames, nextAction, reward, nextFrames, done))

        shouldReset = done
        if len(replayMemory) % 100 == 1:
            print("At step", len(replayMemory), "out of 50000 of population")

    for episode in range(1, M + 1):

        startStep = steps

        lastFrame = env.reset()
        lastFrame = lastFrame * (1.0 / 255)

        fourFrames = collections.deque()
        fourFrames.append(lastFrame)
        fourFrames.append(lastFrame)
        fourFrames.append(lastFrame)
        fourFrames.append(lastFrame)
        done = False

        if render:
            env.render()

        totalReward = 0

        while not done:

            # process lastFrame
            lastFrames = np.array([np.asarray(fourFrames, dtype=np.float32)])

            # pick random action
            nextAction = env.action_space.sample()

            # occurs 1 - epsilon of the time
            if 1 - epsilon >= random.uniform(0, 1):
                # pick action action
                valueVector = qModel.predict(tf.transpose(
                    tf.convert_to_tensor(lastFrames, dtype=tf.float32), perm=[0, 2, 3, 1]), steps=1)
                nextAction = np.argmax(valueVector[0])

            # repeat action every 4 frames like paper
            for i in range(0, random.randint(1,4)):

                frame, reward, done, envInfo = env.step(nextAction)

                totalReward += reward

                #if lives != None and envInfo['ale.lives'] < lives:
                #    done = True
                #elif lives == None:
                #    lives = envInfo['ale.lives']

                frame = frame * (1.0/255)
                fourFrames.popleft()
                fourFrames.append(frame)

                if reward > 0:
                    reward = 1
                elif reward < 0:
                    reward = -1
                else:
                    reward = 0

                if steps % 100 == 0:
                    print("On step", steps, "total reward is", totalReward)
                    gc.collect()
                    print(avgLoss/100.0)
                    avgLoss = 0

                nextFrames = np.array(
                    [np.asarray(fourFrames, dtype=np.float32)])

                if len(replayMemory) == memorySize:
                    entry = replayMemory.popleft()
                    del entry

                replayMemory.append(
                    (lastFrames, nextAction, reward, nextFrames, done))

                if render:
                    env.render()

                steps += 1

                # every C steps reset target action value function
                if steps % cSteps == 0:
                    qHatModel.set_weights(qModel.get_weights())
                if epsilon > stopEpsilon:
                    epsilon = max(epsilon - \
                        (startEpsilon-stopEpsilon) / stepsToReachStopEpsilon, stopEpsilon)

                if done:
                    break

            # processing

            trainingFrames = []

            transitions = random.sample(
                replayMemory, min(len(replayMemory), minibatchSize))

            valueVectorCurr = qModel.predict(
                np.array([np.transpose(lastFrames, (0, 2, 3, 1))[0] for lastFrames, _, _, _, _ in transitions]), steps=1, batch_size=len(transitions))
            valueVectorNext = qHatModel.predict(
                np.array([np.transpose(nextFrames, (0, 2, 3, 1))[0] for _, _, _, nextFrames, _ in transitions]), steps=1, batch_size=len(transitions))

            idx = 0

            for lastFrames, nextAction, reward, nextFrames, done in transitions:
                y = reward  # set to reward
                lastFrameTensor = tf.transpose(
                    tf.convert_to_tensor(lastFrames, dtype=tf.float32), perm=[0, 2, 3, 1])
                if not done:
                    # replace with max
                    y += gamma * np.max(valueVectorNext[idx])
                # gradient descent
                valueVectorCurr[idx][nextAction] = y
                trainingFrames.append(lastFrameTensor[0])
                idx += 1

            trainingFrames = tf.convert_to_tensor(
                trainingFrames, dtype=tf.float32)
            trainingValues = tf.convert_to_tensor(
                np.array(valueVectorCurr), dtype=tf.float32)
            info = qModel.fit(x=trainingFrames, y=trainingValues,
                              epochs=1, batch_size=len(transitions), verbose=0)
            avgLoss += info.history['loss'][0]
            # lossOverTime.append(info.history['loss'])
            del trainingFrames
            del trainingValues
            del transitions
            valueVectorCurr = None
            valueVectorNext = None

        print("Episode:", episode, "Reward:", totalReward,
              "Took:", steps-startStep, "steps", "Ending Epsilon:", epsilon, "at step", steps)
        totalRewardOverTime.append(totalReward)
        if os.path.exists(checkpointReward) and os.path.isfile(checkpointReward):
            shutil.copyfile(checkpointReward, checkpointReward+"cpy")
        if os.path.exists(checkpointFile) and os.path.isfile(checkpointFile):
            shutil.copyfile(checkpointFile, checkpointFile+"cpy")
        np.save(checkpointReward, np.array(totalRewardOverTime))
        qModel.save(checkpointFile)


dqn(8000, env, startStep=0, memorySize=300000)
