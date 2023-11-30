import numpy as np
from sklearn import manifold
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import matplotlib.pyplot as plt


def shift_embedding_to_reference(embeddings, true_ldmrk_locs, reference_idx=0):
    """
    Scale the embeddings such that the embedding at reference idx is
    centered at zero, zero. Then scale the embeddings
    """
    x_ref_a, y_ref_a = embeddings[reference_idx, 0], embeddings[reference_idx, 1]
    x_ref_b, y_ref_b = true_ldmrk_locs[reference_idx, 0], true_ldmrk_locs[reference_idx, 1]
    embeddings += (-np.array((x_ref_a, y_ref_a)) + np.array((x_ref_b, y_ref_b)))

    return embeddings, true_ldmrk_locs



def scale_data_min_max(embeddings, true_ldmrk_locs):
    """
    Scale the 2D embeddings such that the distances between the
    embeddings match the distances between the true landmark locations.
    """

    x_min_a, x_max_a = np.min(embeddings[:, 0]), np.max(embeddings[:, 0])
    y_min_a, y_max_a = np.min(embeddings[:, 1]), np.max(embeddings[:, 1])
    x_min_b, x_max_b = np.min(true_ldmrk_locs[:, 0]), np.max(true_ldmrk_locs[:, 0])
    y_min_b, y_max_b = np.min(true_ldmrk_locs[:, 1]), np.max(true_ldmrk_locs[:, 1])

    # scale the embeddings
    embeddings[:, 0] = (embeddings[:, 0] - x_min_a) / (x_max_a - x_min_a)
    embeddings[:, 1] = (embeddings[:, 1] - y_min_a) / (y_max_a - y_min_a)
    
    # scale the true landmark locations
    true_ldmrk_locs[:, 0] = (true_ldmrk_locs[:, 0] - x_min_b) / (x_max_b - x_min_b)
    true_ldmrk_locs[:, 1] = (true_ldmrk_locs[:, 1] - y_min_b) / (y_max_b - y_min_b)

    return embeddings, true_ldmrk_locs 


def mds_cognitive_mapping(cog_map_dist_matrix, true_ldmrk_locs, seed=1):
    """
    Perform multi-dimensional scaling on the 'cog_map_dist_matrix' 
    which is an NxN matrix containing distances between N landmark 
    vectors. Scale the outputs of MDS such that the distances match
    those of the true distance matrix.
    
    Cite: https://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html
    """

    # metric meds
    mds = manifold.MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        random_state=seed,
        dissimilarity="precomputed",
        n_jobs=1,
        normalized_stress="auto",
    )

    pos = mds.fit(cog_map_dist_matrix).embedding_

    # non-metric mds
    # nmds = manifold.MDS(
    #     n_components=2,
    #     metric=False,
    #     max_iter=3000,
    #     eps=1e-12,
    #     dissimilarity="precomputed",
    #     random_state=seed,
    #     n_jobs=1,
    #     n_init=1,
    #     normalized_stress="auto",
    # )
    # npos = nmds.fit_transform(cog_map_dist_matrix, init=pos)

    # rescale the data
    # pos *= np.sqrt(np.sum((true_ldmrk_locs**2), axis=None)) / np.sqrt(np.sum((pos**2), axis=None))
    # npos *= np.sqrt((true_ldmrk_locs**2).sum()) / np.sqrt((npos**2).sum())
    pos, true_ldmrk_locs = scale_data_min_max(pos, true_ldmrk_locs)
    pos, _, _ = procrustes(true_ldmrk_locs, pos)
    pos, true_ldmrk_locs = shift_embedding_to_reference(pos, true_ldmrk_locs, reference_idx=0)

    # rotate the data
    clf = PCA(n_components=2)
    # true_ldmrk_locs = clf.fit_transform(true_ldmrk_locs)
    # pos = clf.fit_transform(pos)
    # npos = clf.fit_transform(npos)

    # I want a matplotlib scatter plot for X_true, pos, and npos.
    fig = plt.figure(1)
    ax = plt.axes([0., 0., 1., 1.]) 
    s = 100


    plt.scatter(true_ldmrk_locs[:, 0], true_ldmrk_locs[:, 1], color='green', s=s, lw=0, label='True Positions')
    plt.scatter(pos[:, 0], pos[:, 1], color='red', s=s, lw=0, label='MDS')

    # plot the first point for true_ldmrk_locs and pos with a star marker (marker='*')
    plt.scatter(true_ldmrk_locs[0, 0], true_ldmrk_locs[0, 1], color='blue', marker='*', s=s, lw=0, label='True Positions')
    plt.scatter(pos[0, 0], pos[0, 1], color='black', marker='*', s=s, lw=0, label='MDS')

    # plt.scatter(npos[:, 0], npos[:, 1], color='turquoise', s=s, lw=0, label='NMDS')
    plt.legend(scatterpoints=1, loc='best', shadow=False)

    return fig

if __name__ == "__main__":
    # test the function

    # create a true landmark location matrix
    # 10 landmarks, 2 dimensions
    true_ldmrk_locs = np.random.rand(10, 2)

    # create a distane matrix from euclidean distance
    # between the landmark locations
    # first add noise to true landmark locations
    _true_ldmrk_locs = true_ldmrk_locs # + np.random.rand(10, 2)*0.1
    dist_matrix = euclidean_distances(_true_ldmrk_locs)

    # run the function
    fig = mds_cognitive_mapping(dist_matrix, true_ldmrk_locs)
    # save the figure
    fig.savefig('test_mds.png')
    plt.show()



