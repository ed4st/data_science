"""Data Clustering"""

# Author: Edgar Baquero <edgar.baquero@cimat.mx>


import numpy as np
from scipy.stats import multivariate_normal
import random

class Cluster:
    #----------------------------Constructor----------------------------
    def __init__(self, method = 'k_means', k = None, iter = 5, kernel_type = 'gaussian', sigma = 1.5):
        """
        Current class performs data classification under numeric data,
        using distinct algorithms (k_means, GMM_EM, kk_means)
        
        Parameters
        ----------            
        method: str
            Method that performs the classification. It'll be 'k_means' by default.
            You can also choose the 'GMM_EM' method, based on Gaussian Mixture Models
            or the 'kk_means' method, based on manifold learning clustering
            
        k: int
            Number of clusters.
            It'll be None by default.
            
        iter: int
            Number of iterations.
            Number of iterations that we want to compute using k_means method.
            It'll be 5 by default
        
        kernel_type: str
            kernel type: 'gaussian', 'polynomial'.
            kernel type used to compute manifold distances when using kernel 
            k-means ('kk_means') method.
            It'll be 'gaussian' by default
            
        sigma: float
            Gaussian Kernel parametter.
        """
        #------------------Class attributtes------------------
        self.method = method
        self.k = k
        self.iter = iter
        self.kernel_type = kernel_type
        self.sigma = sigma
    #----------------------Class Methods----------------------
    def fit(self, data = None):
        """
        fit the data
        
        Parameters
        ----------
        data : DataFrame, Series, None
            Dataset to classify.
            If data is not set, it will be None by default.
        """
        if data is None:
            self.data = None
            raise TypeError
        else:
            self.data = data
            
    def transform(self):
        """
        transform data using chosen method.
        
        Return
        ----------
        original dataframe with corresponding clusters.
        """
        if self.data is None:
            raise TypeError
        else:
            if self.method == 'k_means':
                return self.__k_means()
            if self.method == 'GMM_EM':
                return self.__GMM()
            if self.method == 'kk_means':
                return self.__kk_means()  
             
    def fit_transform(self, data):
        """
        fit to data, then transform it.
        
        Return
        ----------
        original dataframe with corresponding clusters
        """
        if data is None:
            self.data = None
            raise TypeError
        else:
            self.data = data
            if self.method == 'k_means':
                return self.__k_means()
            if self.method == 'GMM_EM':
                return self.__GMM()
            if self.method == 'kk_means':
                return self.__kk_means()
    #-----------------------------Private methods----------------------------
    
    
    #k-means implementation
    def __k_means(self):
        n,d = self.data.shape
        
        #Selecting k distinct random means based on data to initialize the parameters
        random_row = np.random.randint(low=0, high=n, size = self.k)
        means = [ self.data.iloc[row_index,:] for row_index in random_row ]

        
        #creating an array wich contains the corresponding cluster:
         
        for _ in range(self.iter):
            cluster = []
            for row_ix in range(n):
                #encoding:
                distances = [self.__l2_distance(self.data.iloc[row_ix], mean) for mean in means]
                cluster.append(np.argmin(distances))
            #reloading means:
              
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
    
    #Gaussian Mixture Model implementation for data clustering
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
        for _ in range(self.iter):
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
    
    
    def __initialize_cluster(self):
        input_data = self.data.to_numpy()
        n_cluster = self.k
        list_cluster_member = [[] for i in range(n_cluster)]
        shuffled_data = input_data
        np.random.shuffle(shuffled_data)
        for i in range(0, input_data.shape[0]):
            list_cluster_member[i%n_cluster].append(input_data[i,:])
            
        return list_cluster_member

    def __gaussian_kernel(self, data1, data2, sigma):
        delta =abs(np.subtract(data1, data2))
        l2_distance = (np.square(delta).sum(axis=1))
        result = np.exp(-(l2_distance)/(2*sigma**2))
        return result


    def __third_term(self, cluster_member):
        result = 0
        for i in range(0, cluster_member.shape[0]):
            for j in range(0, cluster_member.shape[0]):
                result = result + self.__gaussian_kernel(cluster_member[i, :], cluster_member[j, :], self.sigma)
        result = result / (cluster_member.shape[0] ** 2)
        return result

    def __second_term(self, dataI, cluster_member):
        result = 0
        for i in range(0, cluster_member.shape[0]):
            result = result + self.__gaussian_kernel(dataI, cluster_member[i,:], self.sigma)
        result = 2 * result / cluster_member.shape[0]
        return result

    def __kk_means(self):
        data = self.data.to_numpy()
        
        init_member = self.__initialize_cluster()
        n_cluster = init_member.__len__()
        #looping until converged
        while(True):
            
            result_cluster = np.ndarray(shape=(data.shape[0], 0))
            #assign data to cluster whose centroid is the closest one
            for i in range(0, n_cluster):#repeat for all cluster
                term3 = self.__third_term(np.asmatrix(init_member[i]))
                matrix_3 = np.repeat(term3, data.shape[0], axis=0); matrix_3 = np.asmatrix(matrix_3)
                matrix_2 = np.ndarray(shape=(0,1))
                for j in range(0, data.shape[0]): #repeat for all data
                    term2 = self.__second_term(data[j,:], np.asmatrix(init_member[i]))
                    matrix_2 = np.concatenate((matrix_2, term2), axis=0)
                matrix_2 = np.asmatrix(matrix_2)
                result_cluster_i = np.add(-1*matrix_2, matrix_3)
                result_cluster =\
                    np.concatenate((result_cluster, result_cluster_i), axis=1)
            kcluster = np.ravel(np.argmin(np.matrix(result_cluster), axis=1))
        
            list_cluster_member = [[] for l in range(self.k)]
            for i in range(0, data.shape[0]):#assign data to cluster regarding cluster matrix
                list_cluster_member[np.asscalar(kcluster[i])].append(data[i,:])
            
            #break when converged
            boolAcc = True
            for m in range(0, n_cluster):
                prev = np.asmatrix(init_member[m])
                current = np.asmatrix(list_cluster_member[m])
                if (prev.shape[0] != current.shape[0]):
                    boolAcc = False
                    break
                if (prev.shape[0] == current.shape[0]):
                    boolPerCluster = (prev == current).all()
                boolAcc = boolAcc and boolPerCluster
                if(boolAcc==False):
                    break
            if(boolAcc==True):
                break
            #iterationCounter += 1
            #update new cluster member
            init_member = list_cluster_member
            #newTime = np.around(time.time(), decimals=0)
            #print("iteration-", iterationCounter, ": ", newTime - oldTime, " seconds")
        
        KKMdf = self.data.copy()
        KKMdf['cluster'] = kcluster
        return KKMdf

        
        
