#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def step(action):
    print(action)
    # if not isinstance(action, list):
    #     action = [action]
    obs, reward, done, info = env.step(action)
    # print('step=%s, reward=%.2f' % (env.step_count, reward))
    print('step=', env.step_count, ', reward=', reward)

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)

def key_handler(event):
    # print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

actions = []
def key_handler2(event):
    # print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        actions.append(env.actions.left)
        if len(actions) == 2:
            step(actions)
            actions.clear()
        return
    if event.key == 'right':
        actions.append(env.actions.right)
        if len(actions) == 2:
            step(actions)
            actions.clear()
        return
    if event.key == 'up':
        actions.append(env.actions.forward)
        if len(actions) == 2:
            step(actions)
            actions.clear()
        return

    # Spacebar
    if event.key == ' ':
        actions.append(env.actions.toggle)
        if len(actions) == 2:
            step(actions)
            actions.clear()
        return
    if event.key == 'pageup':
        actions.append(env.actions.pickup)
        if len(actions) == 2:
            step(actions)
            actions.clear()
        return
    if event.key == 'pagedown':
        actions.append(env.actions.drop)
        if len(actions) == 2:
            step(actions)
            actions.clear()
        return

    if event.key == 'enter':
        actions.append(env.actions.done)
        if len(actions) == 2:
            step(actions)
            actions.clear()
        return

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-MultiAgent-N6-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)

args = parser.parse_args()

env = gym.make(args.env)

if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

env = SingleAgentWrapper(env)
agent_num = len(env.agents)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
