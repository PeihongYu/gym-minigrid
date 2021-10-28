#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import random


def redraw(img):
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


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-MultiAgent-N2-S4-A1R-v0'
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
env = FullyObsWrapper(env)
# env = SingleAgentWrapper(env)
# env = TwoAgentWrapper(env, mode="full")

visualize = False
if visualize:
    window = Window('gym_minigrid - ' + args.env)
    reset()
else:
    env.reset()


for key_method in range(2):

    ep_steps = []
    done_count = 0
    done = False
    lam = {}

    start = time.time()
    while done_count < 1000:
        # action = random.randrange(16)
        # action = random.randrange(4)
        # action = [random.randrange(4), random.randrange(4)]
        action = [random.randrange(4)]
        obs, reward, done, info = env.step(action)
        # print('step=', env.step_count, ', reward=', reward)

        if key_method == 0:
            key = obs['image'].tobytes()
        elif key_method == 1:
            key = tuple(obs['image'].ravel())
        else:
            key = tuple(env.gen_key())

        if lam.get(key):
            lam[key] += 1
        else:
            lam[key] = 1

        if done:
            # print('done!')
            ep_steps.append(env.step_count)
            if visualize:
                reset()
            else:
                env.reset()
            done_count += 1
        else:
            if visualize:
                redraw(obs)
    end = time.time()

    print('key generation method: ', key_method)
    print('ave episode length: ', sum(ep_steps) / len(ep_steps))
    print('time total: ', end - start)
    print('time per 1000 step: ', (end - start) / sum(ep_steps) * 1000)



if visualize:
    # Blocking event loop
    window.show(block=True)
