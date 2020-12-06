
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

gs = gridspec.GridSpec(2,2)

# fig, (ax1, ax2) = plt.subplots(1, 2)
fig = plt.figure()
fig.suptitle("Train")
ax1 = fig.add_subplot(gs[0,0])
scores = []
with open('./trained/scores.txt') as f:
    for line in f:
        scores.append(round(float(line.strip('\n'))))

ax1.set_title("recorded score")
ax1.set_xlabel("episode")
ax1.plot(scores)
# plt.show()

ax2 = fig.add_subplot(gs[0,1])
scores = np.array_split(scores, 500)
temp = []
for score in scores:
    temp.append(np.mean(score))
ax2.set_title('mean socre')
ax2.set_xlabel("episode")
ax2.plot(temp)
# plt.show()

ax3 = fig.add_subplot(gs[1,:])
loss = []
with open('./trained/losses.txt') as f:
    for line in f:
        loss.append(float(line.strip('\n')))
ax3.set_title("loss")
ax3.set_xlabel("episode")
ax3.plot(loss)

plt.show()
