import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
class Cluster:
    def __init__(self, data = None, method = 'k_means', k = None, iter = None):
        """
        Current class performs data classification under numeric data,
        using distinct algorithms (k_means, agglomerative)
        
        Parameters
        ----------
        data : DataFrame, Series, None
            Dataset to classify.
            If data is not set, it will be None by default.
            
        method: 'k_means', 'agglomerative' - str
            Method that performs the classification. It'll be 'k_means' by default    
            
        k: int
            Number of clusters.
            It'll be None by default.
            
        iter:
            Number of iterations.
            Number of iterations that we want to compute using k_means method.
            It'll be None by default
        """
        self.data = data
        self.method = method
        self.k = k
        self.iter = iter
        
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
            if self.method == 'agglomerative':
                return self.__agglomerative()   
             
    def __agglomerative(self):
        pass
    
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
    
#proving data



if __name__ == "__main__":
    #defining 2 remarkable data groups:
    data_example = pd.DataFrame({'x':[0,1,2,1,0,6,7,5,6,8,7],
                             'y':[1,3,2,1,1,9,6,7,5,6,8]})
    
    cluster = Cluster(data = data_example, method = 'k_means', k = 2, iter = 10)
    cluster_data = cluster.fit_transform()
    print(cluster_data)
    sns.scatterplot(data = cluster_data, x = cluster_data.x, y = cluster_data.y,  hue = 'cluster')
    plt.show()
    
