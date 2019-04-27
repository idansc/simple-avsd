from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import argparse
import h5py
import json
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
from scipy.spatial import distance
from torch.autograd import Variable

from sklearn.preprocessing import MinMaxScaler

from torchviz import make_dot, make_dot_from_trace

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle


from skimage import io
from skimage import transform
import skimage.io
''''''
counter = 0
matplotlib.rcParams.update({'font.size': 14})
labels = ["SF-Q-1","SF-Q-se-1"]
line_types = [":","--","-","-"]
baselines = [[41.24, 43.33,48.48],
            [70.45, 74.27,78.75]]

x_all = []
y_all = []
color_map = plt.get_cmap('gist_rainbow')

colors = [color_map(1.0 * k / 2 for k in range(2))][::-1]


plt.figure()
# plt.figure(figsize=(4,5))
for j in range(len(folders)):
    p[counter], = plt.plot(x_all[j], y_all[j], color=colors[j], label=labels[j], lw=4, ls=line_types[j // 3])
    plabel[counter] = labels[j]
    counter += 1
baseline = baselines[i]
# for k in range(len(baseline_labels)):
#     p[counter], = plt.plot([x_all[j][0],x_all[j][-1]], [baseline[k],baseline[k]], color=baseline_colors[k], lw=4, ls=line_types[k], label=baseline_labels[k])
#     plabel[counter] = baseline_labels[k]
#     counter += 1
plt.tight_layout(pad=1.08, h_pad=None, w_pad=None)
plt.xticks(np.arange(1, 6))
plt.xlim(0.9, 5.1)
plt.grid(b=True, which='major', color='k', linestyle=':')
ax = plt.gca()
if (i == 4):
    plt.legend([p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]],
               [plabel[0], plabel[1], plabel[2], plabel[3], plabel[4], plabel[5], plabel[6], plabel[7], plabel[8]],
               loc="best", ncol=3, title="Q-only         Q+I only         Q+I+H", prop={'size': 10})
# ax.set_aspect(1.5)
plt.xlabel("Epoch")
plt.ylabel(metric_labels[i])
plt.savefig(os.path.join(output_folder, "visdial_q2_" + str(metric)) + '.png', bbox_inches='tight')
plt.savefig(os.path.join(output_folder, "visdial_q2_" + str(metric)) + '.pdf', bbox_inches='tight')
