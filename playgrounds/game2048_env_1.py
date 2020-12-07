from gym import spaces

import numpy as np
import gym
import random

from six import StringIO
import sys

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


class Env2048(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, size=4):
        self.size = 4
        self.width = self.size
        self.height = self.size
        self.squares = self.width * self.height
        self.action_set = {UP, DOWN, LEFT, RIGHT}

        # gym Env members
        self.action_space = spaces.Discrete(4)
        # observation
        self.observation_space = spaces.Box(
            low=0, high=2**self.squares, shape=(self.width, self.height), dtype=np.int)

        # initial the game
        self.reset()

    def step(self, action):
        if action not in self.action_set:
            raise Exception('Invalid Action Input')
        # in case users still send action after done without reset
        if self.done:
            raise Exception('Game is over')
        if self.ok_to_move(action):
            reward = self.combine(action)
            self.add_tile()
            self.highest = np.max(self.Matrix)
            self.done = self.is_end()
        else:
        # reward = 0
        # set a negative reward to avoid stuck...
            reward = -10
        # or regard it as vital error and game is over
        self.score += reward
        if self.score < -100:
            self.done = True

        self.info = (self.score, self.available, self.highest)
        return self.Matrix, reward, self.done, self.info

    def reset(self):
        self.Matrix = np.zeros((self.width, self.height), dtype=np.int)
        self.available = list(range(self.squares))
        self.score = 0
        self.done = False
        # add two numbers on the initial board
        self.add_tile()
        self.add_tile()
        self.highest = np.max(self.Matrix)
        return self.Matrix

    def render(self,mode="human"):
        if mode == "ansi":
            outfile = StringIO()
        else:
            sys.stdout
        #print('Score: {}'.format(self.score))
        #print('Highest: {}'.format(self.highest))
        #print(self.Matrix)
        s = 'Score: {}\n'.format(self.score)
        s += 'Highest: {}\n'.format(self.highest())
        npa = np.array(self.Matrix)
        grid = npa.reshape((self.size, self.size))
        s += "{}\n".format(grid)
        outfile.write(s)
        return outfile

    def add_tile(self):
        if not self.available:
            return
        empties = self.empties()
        idx = self.np_random.choice(len(empties))
        if self.np_random.random_sample() < 0.9:
            self.Matrix[idx // self.size][idx % self.size] = 2
        else:
            self.Matrix[idx // self.size][idx % self.size] = 4
        self.available.remove(idx)

    def empties(self):
        """Return a list of tuples of the location of empty squares."""
        empties = list()
        for y in range(self.height):
            for x in range(self.width):
                if self.get(x, y) == 0:
                    empties.append((x, y))
        return empties

    def ok_to_move(self, direction):
        if direction == LEFT:
            for i in range(self.size):
                for j in range(self.size - 1):
                    if (self.Matrix[i][j] == 0 and self.Matrix[i][j + 1] > 0) or \
                    (self.Matrix[i][j] != 0 and self.Matrix[i][j] == self.Matrix[i][j + 1]):
                        return True
        elif direction == RIGHT:
            for i in range(self.size):
                for j in range(self.size - 1):
                    if (self.Matrix[i][j + 1] == 0 and self.Matrix[i][j] > 0) or \
                    (self.Matrix[i][j] != 0 and self.Matrix[i][j] == self.Matrix[i][j + 1]):
                        return True
        elif direction == UP:
            for i in range(self.size):
                for j in range(self.size - 1):
                    if (self.Matrix[j][i] == 0 and self.Matrix[j + 1][i]) or \
                        (self.Matrix[j][i] != 0 and self.Matrix[j][i] == self.Matrix[j + 1][i]):
                        return True
        elif direction == DOWN:
            for i in range(self.size):
                for j in range(self.size - 1):
                    if (self.Matrix[j + 1][i] == 0 and self.Matrix[j][i]) or \
                        (self.Matrix[j][i] != 0 and self.Matrix[j + 1][i] == self.Matrix[j][i]):
                        return True
        return False

    def combine(self, direction):
        reward = 0
        if direction == UP or direction == DOWN:
            for col in range(self.size):
                tiles = [x for x in self.Matrix[:, col] if x]
                self.Matrix[:, col] = 0
                i = 0
                if direction == DOWN:
                    tiles = tiles[::-1]
                fill_index = 0 if direction == UP else self.size - 1
                fill_direction = 1 if direction == UP else -1
                while i < len(tiles):
                    if i < len(tiles) - 1 and tiles[i] == tiles[i + 1]:
                        new_num = tiles[i] + tiles[i + 1]
                        reward += new_num
                        self.Matrix[fill_index, col] = new_num
                        fill_index += fill_direction
                        i += 2
                    else:
                        self.Matrix[fill_index, col] = tiles[i]
                        fill_index += fill_direction
                        i += 1
        else:
            for row in range(self.size):
                tiles = [x for x in self.Matrix[row, :] if x]
                self.Matrix[row, :] = 0
                i = 0
                if direction == RIGHT:
                    tiles = tiles[::-1]
                fill_index = 0 if direction == LEFT else self.size - 1
                fill_direction = 1 if direction == LEFT else -1
                while i < len(tiles):
                    if i < len(tiles) - 1 and tiles[i] == tiles[i + 1]:
                        new_num = tiles[i] + tiles[i + 1]
                        reward += new_num
                        self.Matrix[row, fill_index] = new_num
                        fill_index += fill_direction
                        i += 2
                else:
                    self.Matrix[row, fill_index] = tiles[i]
                    fill_index += fill_direction
                    i += 1
        self.updateavailable()
        return reward

    def updateavailable(self):
        self.available = [
            x for x in range(self.squares)
            if not self.Matrix.flatten()[x // self.size][x % self.size]
    ]

    def is_end(self):
        if self.available:
            return False
        for direction in [UP, DOWN, LEFT, RIGHT]:
            if self._ok_to_move(direction):
                return False
        return True
