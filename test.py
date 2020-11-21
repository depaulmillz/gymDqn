import tensorflow as tf
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

tf.keras.backend.clear_session()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

env = gym.make('StarGunnerNoFrameskip-v4')
env = gym.wrappers.AtariPreprocessing(env)
checkpointFile = './checkpoints/checkpoint.h5'


def test_dqn(M, env, render=True):

    # lossOverTime = []
    totalRewardOverTime = []

    actingCNN = tf.keras.models.load_model(checkpointFile)
    steps = 0
    for episode in range(1, M + 1):

        startStep = steps

        lastFrame = env.reset()

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

            # pick action action
            valueVector = actingCNN.predict(tf.transpose(
                tf.convert_to_tensor(lastFrames, dtype=tf.float32), perm=[0, 2, 3, 1]), steps=1)
            #print(steps, valueVector)
            nextAction = np.argmax(valueVector)
            
            for _ in range(0,4):

                frame, reward, done, _ = env.step(nextAction)

                fourFrames.popleft()
                fourFrames.append(frame)

                totalReward += reward

                if steps % 100 == 0:
                    gc.collect()

                if render:
                    env.render()

                steps += 1

                if done:
                    break


        print("Episode:", episode, "Reward:", totalReward,
              "Took:", steps-startStep, "steps")
        totalRewardOverTime.append(totalReward)


test_dqn(10, env)
