
import matplotlib.pyplot as plt
import numpy as np

scores = []
with open('trained\scores.txt') as f:
    for line in f:
        scores.append(round(float(line.strip('\n'))))

plt.plot(scores)
plt.show()

scores = np.array_split(scores, 500)
temp = []
for score in scores:
    temp.append(np.mean(score))
plt.plot(temp)
plt.show()