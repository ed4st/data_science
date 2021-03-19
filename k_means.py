import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
class Cluster:
    #----------------------------Constructor----------------------------
    def __init__(self, data = None, method = 'k_means', k = None, iter = 5):
        """
        Current class performs data classification under numeric data,
        using distinct algorithms (k_means, GMM_EM, agglomerative)
        
        Parameters
        ----------
        data : DataFrame, Series, None
            Dataset to classify.
            If data is not set, it will be None by default.
            
        method: str
            Method that performs the classification. It'll be 'k_means' by default.
            You can choose also the 'agglomerative' method or the 'GMM_EM' method,
            based on Gaussian Mixture Models
            
        k: int
            Number of clusters.
            It'll be None by default.
            
        iter:
            Number of iterations.
            Number of iterations that we want to compute using k_means method.
            It'll be 5 by default
        """
        #------------------Class attributtes------------------
        self.data = data
        self.method = method
        self.k = k
        self.iter = iter
        
    #----------------------Class Methods----------------------
    def fit_transform(self):
        """
        fit to data, then transform it.
        Return
        ----------
        original dataframe with corresponding clusters
        """
        if self.data is None:
            raise TypeError
        else:
            if self.method == 'k_means':
                return self.__k_means()
            if self.method == 'GMM_EM':
                return self.__GMM()
            if self.method == 'agglomerative':
                return self.__agglomerative()   
    #------------------------------------------------------------------------
    def __agglomerative(self):
        pass
    #------------------------------------------------------------------------
    def __k_means(self):
        n,d = self.data.shape
        
        #first, we set k (the number of clusters) random points to be reloaded in every iteration
        means = [np.random.randint(max(np.max(self.data)), size=2) for i in range(self.k)]
        #means = [np.array([0,0]),np.array([9,9])]
        
        #creating an array wich contains the corresponding cluster:
         
        for iter in range(self.iter):
            cluster = []
            for row_ix in range(n):
                #encoding:
                distances = [self.__l2_distance(self.data.iloc[row_ix], mean) for mean in means]
                cluster.append(np.argmin(distances))
            #reloading means:
            print(cluster)        
            for k in range(self.k):
                #getting indices of the data related to cluster k
                k_ix = [ix for ix, clus in enumerate(cluster) if clus == k]
                #selecting kth cluster's data 
                cluster_k = self.data.iloc[k_ix]
                #reloading means:
                means[k] = np.array(cluster_k.mean(axis = 0))
                
        #returning clusterized dataframe:
        KMdf = self.data.copy()
        KMdf['cluster'] = cluster
        return KMdf
    def __l2_distance(self,x,y):
        """
        Returns the euclidean distance between two points in R^n
        
        Parameters
        ----------
        x, y: np.array
            coordinates points to compute distance
        """
        return np.sqrt(np.dot(x-y,x-y))
    #------------------------------------------------------------------------
    def __GMM(self):
        #initialization of model parameters
        n, m = self.data.shape
        phi = np.full(shape=self.k, fill_value=1/self.k)
        weights = np.full(shape = self.data.shape, fill_value=1/self.k)
        
        #Selecting k distinct random means based on data to initialize the parameters
        random_row = np.random.randint(low=0, high=n, size = self.k)
        mu = [ self.data.iloc[row_index,:] for row_index in random_row ]
        #Initializing sigmas matrices    
        sigma = [ np.cov(self.data.T) for _ in range(self.k) ]
        
        #Iterating model with Expectation-Maximization algorithm
        for iteration in range(self.iter):
            #1.Expectation Step:
            
            
            #predicting probability using initialized mixture model
            likelihood = np.zeros((n, self.k))
            for i in range(self.k):
                distribution = multivariate_normal(mean = mu[i],cov= sigma[i])
                likelihood[:,i] = distribution.pdf(self.data)
            
            #calculating responsability of data
            numerator = likelihood * phi
            denominator = numerator.sum(axis=1)[:, np.newaxis]
            weights = numerator / denominator
            phi = weights.mean(axis=0)
            
            
            #2. Maximization Step:
            
            for i in range(self.k):
                weight = weights[:, [i]]
                total_weight = weight.sum()
                mu[i] = (self.data * weight).sum(axis=0) / total_weight
                sigma[i] = np.cov(self.data.T, 
                                  aweights=(weight/total_weight).flatten(), 
                                  bias=True)
            
        #returning clusterized dataframe:
        cluster = np.argmax(weights, axis = 1)
        GMM_df = self.data.copy()
        GMM_df['cluster'] = cluster
        return GMM_df
    
    
#proving data



if __name__ == "__main__":
    #defining 2 remarkable data groups:
    data_example = pd.DataFrame({'x':[0,1,2,1,0,6,7,5,6,8,7],
                             'y':[1,3,2,1,1,9,6,7,5,6,8]})
    
    #proving example with k-means:
    
    cluster = Cluster(data = data_example, method = 'k_means', k = 2, iter = 10)
    kmeans_clustering = cluster.fit_transform()
    print(kmeans_clustering)
    sns.scatterplot(data = kmeans_clustering, x = kmeans_clustering.x, y = kmeans_clustering.y,  hue = 'cluster')
    #plt.show()
    
    #proving example with GMM:
    cluster_gmm = Cluster(data = data_example, method = 'GMM_EM', k = 2, iter = 100)
    gmm_clustering = cluster_gmm.fit_transform()
    print(gmm_clustering)
    sns.scatterplot(data = gmm_clustering, x = gmm_clustering.x, y = gmm_clustering.y,  hue = 'cluster')
    plt.show()
