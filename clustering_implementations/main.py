from Cluster import Cluster #the created class (use help function for documentation)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles


#playing with data    
if __name__ == "__main__":
     
    #creating ramarkable dataset randomly:
    X, y = make_blobs(
    n_samples=200, n_features=2, centers=3, cluster_std=0.5, shuffle=True,random_state=0)
    
    data_toy = pd.DataFrame(X)
    data_toy.columns = ['x1','x2']
    
    #--------------------clustering with k-means--------------------
    k_means_cluster = Cluster(method = 'k_means', k = 3, iter = 20)
    data_transformed = k_means_cluster.fit_transform(data = data_toy)
    print(f'k-means-clustering: \n{data_transformed}')
    
    #NOTE: Uncomment the next two lines to get the visualization of data.
    #If you are going to instanciate the class with another clustering methods ('GMM_EM' or "kk_means")
    #please comment again cause it might generate overlaping plots.
    
    #sns.scatterplot(data = data_transformed, x = data_transformed.x1, y = data_transformed.x2,  hue = 'cluster')
    #plt.show()
    
    
    
    #--------------------clustering with Gaussian Mixture Models (GMM_EM)--------------------
    GMM_cluster = Cluster(method = 'k_means', k = 3, iter = 20)
    data_transformed = GMM_cluster.fit_transform(data = data_toy)
    print(f'Gaussian Mixture Models clustering: \n{data_transformed}')
    
    #NOTE: Uncomment the next two lines to get the visualization of data.
    #If you are going to instanciate the class with another clustering methods ('GMM_EM' or "kk_means")
    #please comment again cause it might generate overlaping plots.
    
    #sns.scatterplot(data = kmeans_clustering, x = kmeans_clustering.x1, y = kmeans_clustering.x2,  hue = 'cluster')
    #plt.show()
    
    
    
    #--------------------clustering with kernel k-means (kk_means)--------------------
    #first, we create nonlinear data for example purposes:
    X, y = make_circles(n_samples=100, factor=.1, noise=.05)
    data_toy = pd.DataFrame(X)
    data_toy.columns = ['x1','x2']
       
    kk_means_cluster = Cluster(method = 'kk_means', k = 2, iter = 10, sigma=.1)
    data_transformed = kk_means_cluster.fit_transform(data = data_toy)
    print(f'kernel k-means-clustering: \n{data_transformed}')
    
    #NOTE: the same 2 comments above apply here, except that it's uncommented:
    
    sns.scatterplot(data = data_transformed, x = data_transformed.x1, y = data_transformed.x2,  hue = 'cluster')
    plt.show()
    
    
    
