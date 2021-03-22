import numpy as np
from scipy.stats import multivariate_normal

class Cluster:
    #----------------------------Constructor----------------------------
    def __init__(self, method = 'k_means', k = None, iter = 5, kernel_type = 'gaussian'):
        """
        Current class performs data classification under numeric data,
        using distinct algorithms (k_means, GMM_EM, kk_means)
        
        Parameters
        ----------            
        method: str
            Method that performs the classification. It'll be 'k_means' by default.
            You can choose also the 'GMM_EM' method, based on Gaussian Mixture Models
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
        """
        #------------------Class attributtes------------------
        self.method = method
        self.k = k
        self.iter = iter
        self.kernel_type = kernel_type
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
    
    
    #kernel k-means implementation
    def __kk_means(self):
        n,d = self.data.shape
        
        #Selecting k distinct random means based on data to initialize the parameters
        random_row = np.random.randint(low=0, high=n, size = self.k)
        means = [ self.data.iloc[row_index,:] for row_index in random_row ]

        
        #creating an array wich contains the corresponding cluster:
         
        for _ in range(self.iter):
            cluster = []
            for row_ix in range(n):
                #encoding based on kernel type:
                if self.kernel_type == 'gaussian':    
                    distances = [self.__gaussian_distance(self.data.iloc[row_ix], mean) for mean in means]
                if self.kernel_type == 'polynomial':    
                    distances = [self.__polynomial_distance(self.data.iloc[row_ix], mean) for mean in means]
                    
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
        KKMdf = self.data.copy()
        KKMdf['cluster'] = cluster
        return KKMdf
    
    def __gaussian_kernel(self, x, y):
        """
        Returns the kernel related to the two points 
        given by the transformation of gaussian kernel
        
        Parameters
        ----------
        x, y: np.array
            coordinates points to compute the kernel
        """
        #It's important to note that we are fixing sigma = 1.5
        l2_distance = self.__l2_distance(x,y)
        sigma = 1.5
        e = 2.718281828
        return e**(-(l2_distance**2)/(2*(1.5)**2))

    #filling kernel matrix
    def __kernel_matrix(self, method)
        n = self.data.shape[0]
        self.K = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if method == 'gaussian':
                    self.K[i][j] = self.__gaussian_kernel(self.data.iloc[i],self.data.iloc[j])
        

    
    
