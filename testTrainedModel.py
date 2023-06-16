import math
import numpy as np
import random
import keyboard
import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output
from keras.models import load_model
from PIL import Image

from rocketSim import RocketSim

for i in range(10):
    #--------------- Use this if recording------------------------
    # Create a list to store the frames
    frames = []

    # ------------------- Trained Network Control -------------------
    env = RocketSim()
    model = load_model("Trained_Models/2023_06_09-00_32\model_episode_1900_2023_06_09-07_12.h5")
    state = env.reset()
    state_space = 4

    print("Start")
    while True:
        state = np.reshape(state, (1, state_space))
        act_values = model.predict(state, verbose=0)
        action = np.argmax(act_values[0])
        # print(np.argmax(act_values[0]))
        reward, next_state, done = env.step(action)
        state = next_state

        env.update_animation()
        ###################### Use for recording #####################
        # Convert the current figure to an image and add it to the frames list
        frame_path = f"frames/frame_{len(frames)}.png"  # Generate a unique filename for each frame
        env.fig.savefig(frame_path, dpi=80)

        # Add the frame to the frames list
        frames.append(Image.open(frame_path))
        ##############################################################
        if env.done == True:
            break # Stop



    # -------------- Human control -------------
    # rocket = RocketSim()
    # while True:
    #     action = 1
    #     if keyboard.is_pressed("right arrow"):
    #         action = 0
    #     elif keyboard.is_pressed("left arrow"):
    #         action = 2
    #     else:
    #         action = 1

    #     rocket.step(action=action)

    #     rocket.update_animation()

    #     ##################### Use for recording #####################
    #     # Convert the current figure to an image and add it to the frames list
    #     frame_path = f"frames/frame_{len(frames)}.png"  # Generate a unique filename for each frame
    #     rocket.fig.savefig(frame_path, dpi=80)

    #     # Add the frame to the frames list
    #     frames.append(Image.open(frame_path))
    #     #############################################################


    #     # Check for keyboard interrupt to stop the loop
    #     if keyboard.is_pressed("down arrow") or rocket.done == True:
    #         break

    #-------------------------Use for recording------------------------
    plt.close()
    # Save the frames as a GIF
    frames[0].save(f'savedGIFs/1900epochTestFixedTheta{i}.gif', format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=50,
                loop=0,
                optimize=False)
