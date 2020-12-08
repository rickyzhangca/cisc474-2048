
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import helpers

'''
plot the game scores from playing 2048 game with random policy vs trained policy
'''

plt.plot(helpers.read(path='./trained/', name='losses', rounding=False, mode='.txt'))
plt.show()

plt.plot(helpers.means(helpers.read(path='./trained/', name='scores', rounding=False, mode='.txt'),100))
plt.show()

plt.plot(helpers.means(helpers.read(path='./played/', name='final_value_random', rounding=False, mode='.txt'),5))
plt.plot(helpers.means(helpers.read(path='./played/', name='final_value_trained', rounding=False, mode='.txt'),5))
plt.show()