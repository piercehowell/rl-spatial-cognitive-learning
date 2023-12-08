import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import os
import numpy as np
import pandas as pd

def preamble():
    
    sns.set(style="white", font_scale=2.5)

    colors = ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"]
    palette = sns.color_palette(colors, n_colors=5)
    
    sns.set_palette(palette=palette)
    
    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)
    sns.set(rc={"font.size": 16.0})
    x_labels = ax.get_xlabel()
    return fig, ax

def plot_cognitive_map(true_ldmrk_locs, pos, landmark_keys):
    # I want a matplotlib scatter plot for X_true, pos, and npos.
    fig, ax = preamble()

    # ax = plt.axes([-0.5, -0.5, 8.5, 8.5]) 
    s = 100
    # plt.grid()

    plt.scatter(true_ldmrk_locs[:, 1], true_ldmrk_locs[:, 0], color='green', s=s, lw=0, label='True Positions')
    plt.scatter(pos[:, 1], pos[:, 0], color='red', s=s, lw=0, label='MDS Positions')

    # plot the first point for true_ldmrk_locs and pos with a star marker (marker='*')
    plt.scatter(true_ldmrk_locs[0, 1], true_ldmrk_locs[0, 0], color='blue', marker='*', s=s, lw=0, label='True Positions Ref.')
    plt.scatter(pos[0, 1], pos[0, 0], color='black', marker='*', s=s, lw=0, label='MDS Positions Ref.')

    # add the landmarks next to points
    # only for the pos ones
    for i, (x_val, y_val) in enumerate(zip(pos[:,1], pos[:, 0])):
        plt.text(x_val+0.25, y_val-0.25, landmark_keys[i], ha='center', va='center', color='red')
    
    plt.legend(scatterpoints=1, loc='best', shadow=False)
    plt.ylim([-0.5, 8.5])
    plt.xlim([-0.5, 8.5])
    
    # Flip the y-axis
    plt.gca().invert_yaxis()

    img = mpimg.imread('9x9_four_rooms_map.png')
    plt.imshow(img, extent=[-0.5, 8.5, 8.5, -0.5])
    return fig

if __name__ == "__main__":
    A = [[1,1], 
         [2,3],
         [3,6],
         [6,2],
         [6,5],
         [7,7]]
    
    B = [[1.67,1],
         [2.1,3.05],
         [3.8,6.1],
         [6,2],
         [6,5],
         [7,7.5]]
    
    fig = plot_cognitive_map(np.array(A), np.array(B), ['A', 'B', 'C', 'D', 'E', 'F'])
    plt.savefig('plot_test.png')
