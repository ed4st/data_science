class Classifier:
    def __init__(data = None, method = 'k_means', n_clusters = None, iter = None):
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
            
        n_clusters: int
            Number of clusters.
            It'll be None by default.
            
        iter:
            Number of iterations.
            Number of iterations that we want to compute using k_means method.
            It'll be None by default
        """
        self.data = data
        self.method = method
        self.n_clusters = n_clusters
        self.iter = iter
        
    def fit_transform():
        """
        fit to data, then transform it.
        """
        if self.data is None:
            raise TypeError;
        else:
            if self.method == 'k_means':
                return __k_means()
            if self.method == 'agglomerative':
                return __agglomerative()    
    def __agglomerative():
        pass
    def __k_means():
        pass