import random

import numpy as np
cimport numpy as np
import cython

cdef extern from "math.h":
    double abs(double t)


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
    cdef int cost = 0

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
            medoid_centers[curr_class] = cluster_dict[curr_class][np.argmin(optimal_elements)]
        
        # Calculate cost with squared error.
        #dist_cpy = np.copy(dist_matrix)
        #for curr_class in range(k):
        #    cost += np.sum(np.square(dist_cpy[np.ix_(cluster_dict[curr_class],
        #                               cluster_dict[curr_class])]))

        # Check for convergence
        if np.array_equal(medoid_centers, medoid_centers):
            break

    else:
        # Fencepost assign elements to clusters.
        min_elements = np.argmin(dist_matrix[:, medoid_centers], axis=1)
        for curr_class in range(k):
            cluster_dict[curr_class] = np.where(min_elements == curr_class)[0]

    # return results
    return medoid_centers, cluster_dict, cost

@cython.wraparound(False)
@cython.boundscheck(False)
def pairwise_distance(np.ndarray[np.double_t, ndim=1] r):
    cdef int i, j, c, size
    cdef np.ndarray[np.double_t, ndim=1] ans
    size = sum(range(1, r.shape[0] + 1))
    ans = np.empty(size, dtype=r.dtype)

    c = 0
    for i in range(r.shape[0]):
        for j in range(i, r.shape[0]):
            ans[c] = abs(r[i] - r[j])
            c += 1
    return ans

def slow_dot (np.ndarray[np.double_t, ndim=2] A, 
              np.ndarray[np.double_t, ndim=2] B):
    """Low-memory implementation of dot product"""
    R = np.empty([A.shape[0], B.shape[1]])
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            R[i, j] = np.dot(A[i, :], B[:, j])
    return R

def pairwise_euclidean_distances(np.ndarray[np.double_t, ndim=2] x,
                                 np.ndarray[np.double_t, ndim=2] y):
    transp_x = x.T
    return np.sqrt(np.dot(x, transp_x) - 2 * np.dot(y, transp_x) + np.dot(y, y.T))