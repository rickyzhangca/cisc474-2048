
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import helpers

'''
plot the game scores from playing 2048 game with random policy vs trained policy
'''

# Figure 1 - Loss over training for 20000 episodes

plt.plot(helpers.read(path='./trained/', name='losses', rounding=False, mode='.txt'))
plt.show()

# Figure 2 - Total score vs episodes (averaged per 100 episodes)

plt.plot(helpers.means(helpers.read(path='./trained/', name='scores', rounding=False, mode='.txt'),100))
plt.show()

# Figure 3 - Final board value: random policy vs trained policy, averaged per 5 games

plt.plot(helpers.means(helpers.read(path='./played/', name='final_value_random', rounding=False, mode='.txt'),5))
plt.plot(helpers.means(helpers.read(path='./played/', name='final_value_trained', rounding=False, mode='.txt'),5))
plt.show()