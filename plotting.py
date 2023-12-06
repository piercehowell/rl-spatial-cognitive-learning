import matplotlib.pyplot as plt

def plot_cognitive_map(true_ldmrk_locs, pos):
    # I want a matplotlib scatter plot for X_true, pos, and npos.
    fig = plt.figure(1)
    # ax = plt.axes([-0.5, -0.5, 8.5, 8.5]) 
    s = 100
    plt.grid()
    
    plt.scatter(true_ldmrk_locs[:, 0], true_ldmrk_locs[:, 1], color='green', s=s, lw=0, label='True Positions')
    plt.scatter(pos[:, 0], pos[:, 1], color='red', s=s, lw=0, label='MDS')

    # plot the first point for true_ldmrk_locs and pos with a star marker (marker='*')
    plt.scatter(true_ldmrk_locs[0, 0], true_ldmrk_locs[0, 1], color='blue', marker='*', s=s, lw=0, label='True Positions')
    plt.scatter(pos[0, 0], pos[0, 1], color='black', marker='*', s=s, lw=0, label='MDS')
    plt.legend(scatterpoints=1, loc='best', shadow=False)
    plt.ylim([-0.5, 8.5])
    plt.xlim([-0.5, 8.5])
    plt.savefig('test_mds.png')

    return fig