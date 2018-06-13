import numpy as np


class KMeans():
    """Implementation of the KMeans Algorithm"""

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        """constructor

        Args
        ----
            - n_cluster(int): "k" in K-Means
            - max_iter(int): Maximum number of iterations to consider before returning
            - e(double): tolerance. Difference between successive distortions to define "converged"

        """
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''Finds n_cluster clusters in the data x

        We need to determine the optimal cluster memberships and centroids.

        Args
        ----
            - x (numpy array):  N X D input matrix

        Returns
        -------
            - mu(numpy.ndarray): centroids or means
            - r(numpy.array): cluster membership
            - iter(int): number of iterations taken to converge

        Algorithm
        ---------
            - Initialize means by picking self.n_cluster from N data points
            - Update means and membership until convergence

        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        K = self.n_cluster

        # Initialize muk as random datapoints in x.

        point_idx = np.arange(N)
        np.random.shuffle(point_idx)
        cluster_centers = point_idx[:K]  # K x 1
        mu = x[cluster_centers, :]

        # Initializa cost
        J = np.inf

        for i in range(self.max_iter):

            # Compute r
            r = np.zeros(N)
            dist = np.zeros((N, K))

            for n in range(N):
                for k in range(K):
                    dist[n, k] = np.inner(mu[k,:]-x[n,:], mu[k,:]-x[n,:])
                
            r = np.argmin(dist, axis=1)

            J_new = 0
            for n in range(N):
                J_new += dist[n,r[n]]

            J_new /= N # Just computed the "average" distortion to speed up process

            #print("Iteration [",i,"]: J = ", J ," ; Diff = ", np.absolute(J - J_new))
            print("Iteration [",i,"]: J = ", J)

            if np.absolute(J - J_new) <= self.e:
                return (mu, r, i)
            
            J = J_new

            for k in range(K):
                k_idx_samples, = np.where(r == k)
                mu[k] = np.sum(x[k_idx_samples, :], axis=0) / len(k_idx_samples)
 

        print("Did not converge!")
        return (mu, r, self.max_iter)
