import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

def enhance_similarity_scores(emb1, emb2, method='enhanced_cosine'):
    """
    Calculate similarity between embeddings using various methods.
    
    Parameters:
    emb1, emb2: numpy arrays of shape (n_samples, embedding_dim)
    method: str, similarity method to use
    
    Returns:
    float: similarity score between 0 and 1
    """
    
    def enhanced_cosine(e1, e2):
        # L2 normalize embeddings first
        e1_norm = normalize(e1.reshape(1, -1))
        e2_norm = normalize(e2.reshape(1, -1))
        
        # Calculate cosine similarity
        cos_sim = np.matmul(e2_norm, e1_norm.T)
        
        # Apply non-linear transformation to spread out scores
        scaled_sim = np.tanh(2 * cos_sim) * 0.5 + 0.5
        return scaled_sim.max()
    
    def gaussian_kernel(e1, e2, sigma=1.0):
        # Calculate Euclidean distance
        dist = cdist(e1.reshape(1, -1), e2.reshape(1, -1), metric='euclidean')
        # Apply Gaussian kernel
        similarity = np.exp(-dist ** 2 / (2 * sigma ** 2))
        return similarity.max()
    
    def angular_similarity(e1, e2):
        # Convert cosine similarity to angular similarity
        e1_norm = normalize(e1.reshape(1, -1))
        e2_norm = normalize(e2.reshape(1, -1))
        cos_sim = np.matmul(e2_norm, e1_norm.T)
        # Convert to angle (in radians) and normalize to [0,1]
        angular_sim = 1 - np.arccos(np.clip(cos_sim, -1, 1)) / np.pi
        return angular_sim.max()
    
    def combined_similarity(e1, e2):
        # Combine multiple similarity measures
        cos_sim = enhanced_cosine(e1, e2)
        gauss_sim = gaussian_kernel(e1, e2)
        ang_sim = angular_similarity(e1, e2)
        
        # Weight the different similarities
        # You can adjust these weights based on performance
        weights = [0.4, 0.3, 0.3]
        combined = (weights[0] * cos_sim + 
                   weights[1] * gauss_sim + 
                   weights[2] * ang_sim)
        return combined

    # Dictionary of available methods
    methods = {
        'enhanced_cosine': enhanced_cosine,
        'gaussian': gaussian_kernel,
        'angular': angular_similarity,
        'combined': combined_similarity
    }
    
    if method not in methods:
        raise ValueError(f"Method {method} not supported. Choose from {list(methods.keys())}")
    
    return methods[method](emb1, emb2)

def preprocess_embeddings(emb):
    """
    Preprocess embeddings to enhance similarity detection
    """
    # Ensure we're working with numpy array
    emb = emb.numpy() if hasattr(emb, 'numpy') else np.array(emb)
    
    # Apply L2 normalization
    emb_normalized = normalize(emb)
    
    # Optional: Remove low-variance dimensions
    # variance = np.var(emb_normalized, axis=0)
    # mask = variance > np.percentile(variance, 10)
    # emb_filtered = emb_normalized[:, mask]
    
    return emb_normalized

    # return emb




# Results:

# Audio processing:  (24000,) 16000
# Embeddings from audio shape: torch.Size([1, 2048])
# Audio processing:  (24000,) 16000
# Embeddings from audio2 shape: torch.Size([1, 2048])
# Trial with audio files audio2.wav and audio2_twin.wav
# Old Cosine Similarity score: 0.7658122
# Enhanced Similarity Cosine: 0.8792505
# Enhanced Similarity Gaussian: 0.15358484813977985
# Enhanced Similarity Angular: 0.46025136
# Enhanced Similarity Combined: 0.6084176441617999

# Audio processing:  (24000,) 16000
# Embeddings from audio shape: torch.Size([1, 2048])
# Audio processing:  (24000,) 16000
# Embeddings from audio2 shape: torch.Size([1, 2048])
# Trial with audio files audio.wav and audio2_twin.wav
# Old Cosine Similarity score: 0.58687246
# Enhanced Similarity Cosine: 0.5
# Enhanced Similarity Gaussian: 0.03669846499042128
# Enhanced Similarity Angular: 0.30867672
# Enhanced Similarity Combined: 0.3499427133680908

# Audio processing:  (24000,) 16000
# Embeddings from audio shape: torch.Size([1, 2048])
# Audio processing:  (24000,) 16000
# Embeddings from audio2 shape: torch.Size([1, 2048])
# Trial with audio files audio.wav and audio_twin.wav
# Old Cosine Similarity score: 0.6059735
# Enhanced Similarity Cosine: 0.5
# Enhanced Similarity Gaussian: 0.042757439528585475
# Enhanced Similarity Angular: 0.32260182
# Enhanced Similarity Combined: 0.35533204136583824

# Audio processing:  (24000,) 16000
# Embeddings from audio shape: torch.Size([1, 2048])
# Audio processing:  (24000,) 16000
# Embeddings from audio2 shape: torch.Size([1, 2048])
# Trial with audio files audio.wav and audio2_twin.wav
# Old Cosine Similarity score: 0.58687246
# Enhanced Similarity Cosine: 0.5
# Enhanced Similarity Gaussian: 0.03669846499042128
# Enhanced Similarity Angular: 0.30867672
# Enhanced Similarity Combined: 0.3499427133680908
