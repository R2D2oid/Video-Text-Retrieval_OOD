from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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

def tsne_dim_reduction(embeddings):
	'''
	Input: 
	embeddings: np.array of shape num_samples x embedding_size
	Output:
	np.array of shape num_samples x 2
	'''
	return TSNE(random_state = 42).fit_transform(embeddings)


