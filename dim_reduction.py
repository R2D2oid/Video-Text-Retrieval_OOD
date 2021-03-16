from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np

# calculating tsne for high dimensional vectors is very slow and it could run out of memory.
# to make it faster, try lowering the dimensionality to a reasonable size with PCA then use tsne.

def pca_dim_reduction(embeddings, pca_dims = 50):
    '''
    Input: 
    embeddings: np.array of shape num_samples x embedding_size
    pca_dims: the target dimensionality size. default is 50
    Output:
    np.array of shape num_samples x pca_dims
    '''
    pca = PCA(n_components = pca_dims)
    pca_model = pca.fit(embeddings)
    return pca_model.transform(embeddings)


def pca_dim_reduction_2d(embeddings, pca_dims = 50):
    '''
    Input: 
    embeddings: np.array of shape num_samples x embedding_size x T
    pca_dims: the target dimensionality size. default is 50
    Output:
    np.array of shape num_samples x pca_dims x T
    '''
    pca = PCA(n_components = pca_dims)
    embeddings = np.dstack(embeddings)
    embeddings = np.swapaxes(embeddings, 0, 2) # swap n_vids and T (ex. n_vids=100, T=5)

    dims = embeddings.shape 
    n_samples = dims[0] # maintains
    n_feats = dims[1]   # reduce this from n_feats -> pca_dims
    T = dims[2]         # maintains but makes sure the reduction happen
    
    embeddings_reduced = np.zeros([n_samples, pca_dims, T]) # n_vids, n_feats , T (ex. 100,512,5)
    for t in range(T):
        pca_model = pca.fit(embeddings[:,:,t])
        embeddings_reduced[:,:,t] = pca_model.transform(embeddings[:,:,t])
    
    return embeddings_reduced
    
    
def tsne_dim_reduction(embeddings):
    '''
    Input: 
    embeddings: np.array of shape num_samples x embedding_size
    Output:
    np.array of shape num_samples x 2
    '''
    return TSNE(random_state = 42).fit_transform(embeddings)


