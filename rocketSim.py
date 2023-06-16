#%%
import math
import numpy as np
import random
import keyboard


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import clear_output
from keras.models import load_model

#%%
def lin_normalize_value(value, max_expected_val):
        normalized_value = (value - (-max_expected_val)) / (max_expected_val - (-max_expected_val))
        normalized_value -= 0.5
        return normalized_value

def sig_val(x):
    return (1 / (1 + math.exp(-x))) - 0.5

class RocketSim():

    def __init__(self):

        # Reward Variables
        self.done = False
        self.reward = 0
        self.time_cutoff = 0.75

        ########## DELETE THIS############
        self.score_tracker = 0
        ##############################

        # Rocket dimensions
        self.rocket_width = 0.1
        self.rocket_height = 1  # Height of rocket (m)

        # Constants
        self.g = 9.81  # Acceleration due to gravity (m/s^2)
        self.dt = 0.01  # Time step (s)
        self.m = 0.025  # Mass of the rocket (1-kg)
        # self.A = self.rocket_height * self.rocket_width  # Cross-sectional area of the rocket (m^2)
        # self.Cd = 0.1  # Drag coefficient of the rocket
        # self.rho = 1.2  # Density of air (kg/m^3)

        # Initial conditions
        self.t = 0
        self.x = 0 # Initial position (m)
        self.y = 0  # Initial position (m)
        self.vx = 0  # Initial velocity (m/s)
        self.vy = 0  # Initial velocity (m/s)
        self.rocket_theta = np.deg2rad(random.uniform(-360, 360))  # Initial angle of rocket (rad)
        self.omega = np.deg2rad(random.uniform(-500*math.pi, 500*math.pi))  # Initial angular velocity (rad/s)
        self.F_thrust = 2  # Thrust force (N)
        self.thrust_angle = np.deg2rad(0)  # Thrust angle relative to rocket angle
        self.flipTheta = 15

        # Moment of inertia of the rocket
        self.I = self.m * (self.rocket_width**2 + self.rocket_height**2) / 12

        # Initial Torques
        self.torqueThrust = -self.F_thrust*math.sin(self.thrust_angle) * self.rocket_height
        tau_net = self.torqueThrust
        self.alpha = tau_net / self.I

        # Thrust vector components
        self.F_thrust_x = self.F_thrust * math.sin(self.rocket_theta - self.thrust_angle)
        self.F_thrust_y = self.F_thrust * math.cos(self.rocket_theta - self.thrust_angle)

        # Animation Initialization rocket_width
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.axis('square')
        self.ax.grid(True, linestyle='-')
        # self.ax.axvline(color="purple", linewidth=4)

        self.ax.set_xlabel('Distance (m)')
        self.ax.set_ylabel('Height (m)')

        # Create the rocket vertices
        self.rect_vertices = np.array([( self.x -self.rocket_width/2*math.cos(self.rocket_theta) +self.rocket_height/2*math.sin(self.rocket_theta), self.y -self.rocket_width/2*math.sin(self.rocket_theta) - self.rocket_height/2*math.cos(self.rocket_theta)),
                            ( self.x +self.rocket_width/2*math.cos(self.rocket_theta) + self.rocket_height/2*math.sin(self.rocket_theta), self.y +self.rocket_width/2*math.sin(self.rocket_theta) - self.rocket_height/2*math.cos(self.rocket_theta)),
                            ( self.x +self.rocket_width/2*math.cos(self.rocket_theta) - self.rocket_height/2*math.sin(self.rocket_theta), self.y +self.rocket_width/2*math.sin(self.rocket_theta) + self.rocket_height/2*math.cos(self.rocket_theta)),
                            ( self.x -self.rocket_width/2*math.cos(self.rocket_theta) - self.rocket_height/2*math.sin(self.rocket_theta), self.y -self.rocket_width/2*math.sin(self.rocket_theta) + self.rocket_height/2*math.cos(self.rocket_theta))])

        # Create the thrust vertices
        self.tri_vertices = np.array([( self.x -self.rocket_width/2*math.cos(self.rocket_theta) + self.rocket_height/2*math.sin(self.rocket_theta), self.y -self.rocket_width/2*math.sin(self.rocket_theta) - self.rocket_height/2*math.cos(self.rocket_theta)),
                            ( self.x +self.rocket_width/2*math.cos(self.rocket_theta) + self.rocket_height/2*math.sin(self.rocket_theta), self.y +self.rocket_width/2*math.sin(self.rocket_theta) - self.rocket_height/2*math.cos(self.rocket_theta)),
                            ( self.x + self.rocket_height/2*math.sin(self.rocket_theta) - self.rocket_height/2*math.sin(self.rocket_theta + self.thrust_angle - math.pi), \
                                self.y -self.rocket_width/2*math.sin(self.rocket_theta) - self.rocket_height/2*math.cos(self.rocket_theta) + self.rocket_height/2*math.cos(self.rocket_theta + self.thrust_angle - math.pi))])

        # Create the patches
        self.rocket_patch = self.ax.fill(self.rect_vertices[:,0], self.rect_vertices[:,1], 'b')[0]
        self.thrust_patch = self.ax.fill(self.tri_vertices[:,0], self.tri_vertices[:,1], 'r')[0]


    def adjust_anglePos(self):
        """
        Adjusts the relative angle of the thrust by +1.5 degree.
        The angle stays within 45 and -45 degrees.
        """
        new_angle = np.rad2deg(self.thrust_angle) + 1.5
        if new_angle > 45:
            self.thrust_angle = np.deg2rad(45)
        else:
            self.thrust_angle = np.deg2rad(new_angle)

    def adjust_angleNeg(self):
        """
        Adjusts the relative angle of the thrust by -1.5 degree.
        The angle stays within 45 and -45 degrees.
        """
        new_angle = np.rad2deg(self.thrust_angle) - 1.5
        if new_angle < -45:
            self.thrust_angle = np.deg2rad(-45)
        else:
            self.thrust_angle = np.deg2rad(new_angle)


    def euler(self):
        # Force of Gravity
        F_gravity = self.m * self.g

        # # Air resistance force
        # v = np.array([self.vx, self.vy])
        # v_mag = np.linalg.norm(v)
        # v_hat = v / v_mag if v_mag > 0 else np.array([0, 0])
        # F_drag = -0.1 * self.Cd * self.rho * self.A * v_mag**2 * v_hat

        # Thrust vector components
        F_thrust_x = -self.F_thrust * math.sin(self.rocket_theta - self.thrust_angle)
        F_thrust_y = self.F_thrust * math.cos(self.rocket_theta - self.thrust_angle)

        F_net_x = F_thrust_x #+ F_drag[0]
        F_net_y = F_thrust_y - F_gravity #+ F_drag[1]

        ax = F_net_x / self.m
        ay = F_net_y / self.m

        # Torques
        # cp = np.array([self.x, self.y]) + np.array([self.rocket_height/2*math.sin(self.rocket_theta), -self.rocket_height/2*math.cos(self.rocket_theta)])
        # r = cp - np.array([self.x, self.y])
        # F_drag_torque = np.cross(r, F_drag)
        torqueThrust = -self.F_thrust * math.sin(self.thrust_angle) * self.rocket_height
        tau_net = torqueThrust #+ F_drag_torque

        # Angular acceleration due to thrust
        self.alpha = tau_net / self.I

        # Agular velocity due to angular acceleration
        self.omega += self.alpha * self.dt
        self.rocket_theta += self.omega * self.dt  # Update rocket angle

        # Adjust rocket theta if it exceeds the range
        self.rocket_theta = math.fmod(self.rocket_theta, 2 * math.pi)

        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        self.vx += ax * self.dt
        self.vy += ay * self.dt

        self.t += self.dt

    def update_animation(self):
        self.rocket_patch.remove()
        self.thrust_patch.remove()

        # Create the rocket vertices
        self.rect_vertices = np.array([( self.x -self.rocket_width/2*math.cos(self.rocket_theta) +self.rocket_height/2*math.sin(self.rocket_theta), self.y -self.rocket_width/2*math.sin(self.rocket_theta) - self.rocket_height/2*math.cos(self.rocket_theta)),
                            ( self.x +self.rocket_width/2*math.cos(self.rocket_theta) + self.rocket_height/2*math.sin(self.rocket_theta), self.y +self.rocket_width/2*math.sin(self.rocket_theta) - self.rocket_height/2*math.cos(self.rocket_theta)),
                            ( self.x +self.rocket_width/2*math.cos(self.rocket_theta) - self.rocket_height/2*math.sin(self.rocket_theta), self.y +self.rocket_width/2*math.sin(self.rocket_theta) + self.rocket_height/2*math.cos(self.rocket_theta)),
                            ( self.x -self.rocket_width/2*math.cos(self.rocket_theta) - self.rocket_height/2*math.sin(self.rocket_theta), self.y -self.rocket_width/2*math.sin(self.rocket_theta) + self.rocket_height/2*math.cos(self.rocket_theta))])

        # Create the thrust vertices
        self.tri_vertices = np.array([( self.x -self.rocket_width/2*math.cos(self.rocket_theta) + self.rocket_height/2*math.sin(self.rocket_theta), self.y -self.rocket_width/2*math.sin(self.rocket_theta) - self.rocket_height/2*math.cos(self.rocket_theta)),
                            ( self.x +self.rocket_width/2*math.cos(self.rocket_theta) + self.rocket_height/2*math.sin(self.rocket_theta), self.y +self.rocket_width/2*math.sin(self.rocket_theta) - self.rocket_height/2*math.cos(self.rocket_theta)),
                            ( self.x + self.rocket_height/2*math.sin(self.rocket_theta) - self.rocket_height/2*math.sin(self.rocket_theta + self.thrust_angle - math.pi), \
                                self.y -self.rocket_width/2*math.sin(self.rocket_theta) - self.rocket_height/2*math.cos(self.rocket_theta) + self.rocket_height/2*math.cos(self.rocket_theta + self.thrust_angle - math.pi))])

        # Create the patches
        self.rocket_patch = self.ax.fill(self.rect_vertices[:,0], self.rect_vertices[:,1], 'b')[0]
        self.thrust_patch = self.ax.fill(self.tri_vertices[:,0], self.tri_vertices[:,1], 'r')[0]

        # Set the limits of the plot centered on the rocket
        self.ax.set_xlim([self.x - 15*self.rocket_width, self.x + 15*self.rocket_width])
        self.ax.set_ylim([self.y - 2*self.rocket_height, self.y + 2*self.rocket_height])

        self.score_tracker+=self.reward
        self.ax.set_title('Rocket Simulation - t: {:.2f}, theta: {:.2f}π, omega: {:.2f}pi, thrust_angle: {:.2f}, reward: {:.2f}'\
                .format(self.t, self.rocket_theta/math.pi, self.omega/math.pi, np.rad2deg(self.thrust_angle), self.score_tracker))

        # Pause the program for a short time to slow down the animation
        plt.pause(0.0001)

    def reset(self):
        # Initial conditions
        self.reward = 0
        self.t = 0
        self.x = 0 # Initial position (m)
        self.y = 0  # Initial position (m)
        self.vx = 0  # Initial velocity (m/s)
        self.vy = 0  # Initial velocity (m/s)
        self.rocket_theta = np.deg2rad(random.uniform(-360, 360))  # Initial angle of rocket (rad)
        self.omega = np.deg2rad(random.uniform(-500*math.pi, 500*math.pi))  # Initial angular velocity (rad/s)
        self.F_thrust = 2  # Thrust force (N)
        self.thrust_angle = np.deg2rad(0)  # Thrust angle relative to rocket angle
        self.flipTheta = 15

        # Moment of inertia of the rocket
        self.I = self.m * (self.rocket_width**2 + self.rocket_height**2) / 12

        # Initial Torques
        self.torqueThrust = -self.F_thrust*math.sin(self.thrust_angle) * self.rocket_height

        # Thrust vector components
        self.F_thrust_x = self.F_thrust * math.sin(self.rocket_theta - self.thrust_angle)
        self.F_thrust_y = self.F_thrust * math.cos(self.rocket_theta - self.thrust_angle)
        self.done = False
        # return [lin_normalize_value(self.rocket_theta, 2*math.pi), lin_normalize_value(self.omega, 500),
        #         lin_normalize_value(self.alpha, 500), lin_normalize_value(self.thrust_angle, 45)]
        return [math.sin(self.rocket_theta), lin_normalize_value(self.omega, 500),
                lin_normalize_value(self.alpha, 500), lin_normalize_value(self.thrust_angle, 45)]
        # return [self.rocket_theta, self.omega, self.alpha, self.thrust_angle]

    def step(self, action):
        # Update Euler's method
        self.euler()

        # Perform the action
        if action == 0:
            self.adjust_anglePos()
        elif action == 2:
            self.adjust_angleNeg()

        # Define the target angle and range of stability
        target_angle = 0
        stability_range = 0.0174533  # 1 degree in radians

        # Reward stability within the target range
        # if self.rocket_theta >= 0:
        #     if self.rocket_theta >= (2*math.pi - stability_range) or self.rocket_theta <= stability_range:
        #         stability_reward = 0.2
        #     else:
        #         stability_reward = abs(math.sin(self.rocket_theta/2))**2 *-0.2
        # else:
        #     if self.rocket_theta <= (-2*math.pi + stability_range) or self.rocket_theta >= -stability_range:
        #         stability_reward = 0.2
        #     else:
        stability_reward = 0
        stability_reward += abs(math.sin(self.rocket_theta/2)) *-0.4+0.2
        if self.rocket_theta >= 0 and self.rocket_theta <= stability_range:
            stability_reward += 0.1
        elif self.rocket_theta <= 0 and self.rocket_theta >= -stability_range:
            stability_reward += 0.1
        elif self.rocket_theta <= 2*math.pi and self.rocket_theta >= 2*math.pi - stability_range:
            stability_reward += 0.1
        elif self.rocket_theta >= -2*math.pi and self.rocket_theta <= -2*math.pi + stability_range:
            stability_reward += 0.1

        # Combine the deviation penalty and stability reward
        self.reward = stability_reward

        # Check if simulation is finished
        if self.t >= self.time_cutoff:
            self.done = True

        # Create state vector (dimensions = 6)
        # state = [lin_normalize_value(self.rocket_theta, 2*math.pi), lin_normalize_value(self.omega, 500),
        #         lin_normalize_value(self.alpha, 500), lin_normalize_value(self.thrust_angle, 45)]
        state = [math.sin(self.rocket_theta), lin_normalize_value(self.omega, 500),
                lin_normalize_value(self.alpha, 500), lin_normalize_value(self.thrust_angle, 45)]
        # state = [self.rocket_theta, self.omega, self.alpha, self.thrust_angle]

        return self.reward, state, self.done



# %%

