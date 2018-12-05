import numpy as np
import random


def k_medoids(dist_matrix, int k, iter_max=100):
    m, n = dist_matrix.shape

    if k > n:
        raise Exception("More clusters than datapoints.")

    # Prepare to throw stuff out.
    valid_med_idx = set(range(n))
    invalid_med_idx = set([])

    # Deal with the zero-distance cases.
    dup_rows, dup_cols = np.where(dist_matrix == 0)

    shuffled_idx = list(range(len(dup_rows)))
    np.random.shuffle(shuffled_idx)
    dup_rows = dup_rows[shuffled_idx]
    dup_cols = dup_cols[shuffled_idx]

    for row_val, col_val in zip(dup_rows, dup_cols):
        # Handle if 2+ points have a dist. of 0 by keeping the first one.
        # We'll use it for cluster initialization.
        if row_val < col_val and row_val not in invalid_med_idx:
            invalid_med_idx.add(col_val)

    # Remove the invalid indices from our valid list.
    valid_med_idx = list(valid_med_idx - invalid_med_idx)

    if k > len(valid_med_idx):
        raise Exception("Encountered {} duplicates, now there are more clusters\
                         than datapoints." % len(valid_med_idx))

    # Randomly initialize medoids.
    medoid_centers = np.array(valid_med_idx)
    np.random.shuffle(medoid_centers)
    medoid_centers = np.sort(medoid_centers[:k])

    cluster_dict = {}
    med_copy = np.copy(medoid_centers)

    cost = 0

    while (iter_max):
        iter_max -= 1

        # Assign elements to a cluster.
        min_elements = np.argmin(dist_matrix[:, medoid_centers], axis=1)
        for curr_class in range(k):
            cluster_dict[curr_class] = np.where(min_elements == curr_class)[0]

        # Move medioids.
        for curr_class in range(k):
            optimal_elements = np.mean(dist_matrix[np.ix_(cluster_dict[curr_class],
                                       cluster_dict[curr_class])], axis=1)
            med_copy[curr_class] = cluster_dict[curr_class][np.argmin(optimal_elements)]
        
        # Calculate cost with squared error.
        dist_cpy = np.copy(dist_matrix)
        for curr_class in range(k):
            cost += np.sum(np.square(dist_cpy[np.ix_(cluster_dict[curr_class],
                                       cluster_dict[curr_class])]))

        # Check for convergence
        if np.array_equal(medoid_centers, med_copy):
            break

        medoid_centers = np.copy(med_copy)
    else:
        # Fencepost assign elements to clusters.
        min_elements = np.argmin(dist_matrix[:, medoid_centers], axis=1)
        for curr_class in range(k):
            cluster_dict[curr_class] = np.where(min_elements == curr_class)[0]

    # return results
    return medoid_centers, cluster_dict, cost