# Adapted from https://github.com/phlippe/CITRIS/blob/main/data_generation/data_generation_interventional_pong.py

import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import animation

FRAME_SCALE = 32 # Width = height of frame (pixels)

def get_abs_pos(rel_pos, width):
    assert -1 <= rel_pos <= 1
    abs_pos = width + (FRAME_SCALE-2*width) * (0.5 * rel_pos + 0.5)
    abs_pos = int(abs_pos)
    abs_pos = max(width, min(abs_pos, FRAME_SCALE-width))
    return abs_pos

def mod_pos(pos):
    if pos > 1:
        pos = 2 - pos
    if pos < -1:
        pos = -2 - pos
    return pos

def mod_angle(angle):
    while angle < 0.0:
        angle += 2*np.pi
    while angle > 2*np.pi:
        angle -= 2*np.pi
    return angle

def flip_angle_x(angle):
    angle = mod_angle(np.pi - angle)
    return angle

def flip_angle_y(angle):
    angle = mod_angle(angle + 0.5*np.pi)
    angle = mod_angle(np.pi - angle)
    angle = mod_angle(angle - 0.5*np.pi)
    return angle

class PongState:
    def __init__(
        self,
        ball_x_0       = 0, # Initial x-coordinate of ball               (in [-1, 1])
        ball_y_0       = 0, # Initial y-coordinate of ball               (in [-1, 1])
        ball_vel_0     = np.pi/7, # Initial velocity of ball                   (in [0, 2\pi])
        pro_paddle_y_0 = 0, # Initial y-coordinate of protagonist paddle (in [-1, 1])
        ant_paddle_y_0 = 0, # Initial y-coordinate of antagonist paddle  (in [-1, 1])
        pro_score_0    = 0, # Initial score of protagonist               (integer in [0, 4])
        ant_score_0    = 0, # Initial score of antagonist                (integer in [0, 4])
        ball_speed     = 0.2, # Speed of ball (in [-1, 1])
        ball_noise     = 0.05,
        paddle_noise   = 0.1,
        ball_color     = (0, 0, 255), # RGB color of ball
        background_color = (255, 255, 255), # RGB color of background
        pro_color = (0, 255, 0),
        ant_color = (255, 0, 0)
    ):
        # State variables which are updated every timestep
        self.ball_x       = ball_x_0
        self.ball_y       = ball_y_0
        self.ball_vel     = ball_vel_0
        self.ball_speed   = ball_speed
        self.pro_paddle_y = pro_paddle_y_0
        self.ant_paddle_y = ant_paddle_y_0
        self.pro_score    = pro_score_0
        self.ant_score    = ant_score_0
        self.ball_noise   = ball_noise
        self.paddle_noise = paddle_noise
        self.timestep     = 0
        
        # Style settings
        self.ball_color = np.array(ball_color, dtype=np.uint8)
        self.background_color = np.array(background_color, dtype=np.uint8)
        self.pro_color = np.array(pro_color, dtype=np.uint8)
        self.ant_color = np.array(ant_color, dtype=np.uint8)
    
    def reset_frame(self):
        self.frame = np.ones((FRAME_SCALE, FRAME_SCALE, 3), dtype=np.uint8)
        self.frame *= self.background_color[np.newaxis, np.newaxis, :]
    
    def update_ball_pos(self):
        delta_x = np.cos(self.ball_vel) * self.ball_speed
        delta_y = np.sin(self.ball_vel) * self.ball_speed
        self.ball_x = self.ball_x + delta_x + self.ball_noise*np.random.randn()
        self.ball_y = self.ball_y + delta_y + self.ball_noise*np.random.randn()
        self.ball_vel = mod_angle(self.ball_vel + self.ball_noise*np.random.randn())
        if not(-1 <= self.ball_x <= 1):
            self.ball_x = mod_pos(self.ball_x)
            self.ball_vel = flip_angle_x(self.ball_vel)
        if not(-1 <= self.ball_y <= 1):
            self.ball_y = mod_pos(self.ball_y)
            self.ball_vel = flip_angle_y(self.ball_vel)
    
    def update_paddle_pos(self):
        pro_delta_y = 0.5 * (self.ball_y - self.pro_paddle_y)
        self.pro_paddle_y = self.pro_paddle_y + pro_delta_y + self.paddle_noise*np.random.randn()
        self.pro_paddle_y = max(-1, min(self.pro_paddle_y, 1))
        ant_delta_y = 0.5 * (self.ball_y - self.ant_paddle_y)
        self.ant_paddle_y = self.ant_paddle_y + ant_delta_y + self.paddle_noise*np.random.randn()
        self.ant_paddle_y = max(-1, min(self.ant_paddle_y, 1))
    
    def simulate_timestep(self):
        self.update_ball_pos()
        self.update_paddle_pos()
        self.timestep += 1
    
    def draw_ball(self):
        x = get_abs_pos(self.ball_x, 1)
        y = get_abs_pos(self.ball_y, 1)
        self.frame[x-1:x+2, y, :] = self.ball_color
        self.frame[x, y-1:y+2, :] = self.ball_color
    
    def draw_paddles(self):
        pro_y = get_abs_pos(self.pro_paddle_y, 2)
        self.frame[:2, pro_y-2:pro_y+3, :] = self.pro_color
        ant_y = get_abs_pos(self.ant_paddle_y, 2)
        self.frame[FRAME_SCALE-2:, ant_y-2:ant_y+3, :] = self.ant_color
            
    def draw_frame(self):
        self.reset_frame()
        self.draw_ball()
        self.draw_paddles()
    
    def save_trajectory(self, dest, timesteps=1000, use_progress_bar=False):
        if use_progress_bar:
            progress_bar = tqdm(total=timesteps)
        frames = np.empty((timesteps, FRAME_SCALE, FRAME_SCALE, 3), dtype=np.uint8)
        self.draw_frame()
        frames[0, ...] = self.frame
        for t in range(1, timesteps):
            self.simulate_timestep()
            self.draw_frame()
            frames[t, ...] = self.frame
            if use_progress_bar:
                progress_bar.update(1)
        with open(dest, 'wb') as f:
            np.save(f, frames)
    
    def animate_trajectory(self, src, dest, use_progress_bar=False):
        with open(src, 'rb') as f:
            frames = np.load(f)
        if use_progress_bar:
            progress_bar = tqdm(total=frames.shape[0])
        fig, _ = plt.subplots(figsize=(6, 6))
        def update_fig(t):
            assert t < frames.shape[0]
            frame = frames[t]
            fig.clear()
            ax = fig.gca()
            ax.imshow(frame)
            ax.set_title(f'Timestep: {t}')
            if use_progress_bar:
                progress_bar.update(1)
        anim = animation.FuncAnimation(fig, update_fig, frames.shape[0])
        anim.save(dest, fps=60)