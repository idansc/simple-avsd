from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import os


if __name__ == '__main__':
    color_map = plt.get_cmap('gist_rainbow')
    print([(1.0 * k / 2.0) for k in range(2)])
    colors = [color_map(np.array([1.0 * k / 3.0 for k in range(2)], dtype='float64'))][::-1]
    colors = colors[0]
    plt.figure()
    #15.79, 14.98, 14.48, 14.19
    baseline = [35.36, 22.4873, 19.05, 17.09, 15.79, 14.98, 14.48, 14.19]
    ours = [16.75, 13.70, 12.59, 12.09]
    #no_atten = [17.09, 13.94, 12.79, 12.18]
    o, = plt.plot(range(len(ours)), ours,marker="P", color="red", label="Ours", lw=4, ls="-")
    b, = plt.plot(range(len(baseline)), baseline, marker="o", color="blue", label="Baseline", lw=4, ls="-")
    #n_a, = plt.plot(range(len(no_atten)), no_atten, color=colors[3], label="Baseline", lw=4, ls="--")

    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None)
    plt.xticks(np.arange(0, 8))
    plt.xlim(0, 7)
    plt.ylim(12, 35.36)
    plt.grid(b=True, which='major', color='k', linestyle=':')
    #ax = plt.gca()
    first_legend = plt.legend(handles=[b, o])
    ax = plt.gca().add_artist(first_legend)
    #plt.legend(handles=[o])

    # plt.legend([o, b], ["Ours", "Baseline"])
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.savefig(os.path.join(".", "plot_res") + '.pdf', bbox_inches='tight')
