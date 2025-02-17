import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import norm
import torch
import torch.nn.functional as F
from Detection import ONNXtoTorchModel

class EnhancedSimilarityMatcher:
    def __init__(self, positive_embeddings, negative_embeddings=None, noise_levels=None):
        """
        Initialize with positive and optional negative example embeddings
        """
        # Convert embeddings to numpy and ensure correct shape
        self.positive_embeddings = np.array([emb.squeeze().cpu().numpy() for emb in positive_embeddings])
        self.negative_embeddings = np.array([emb.squeeze().cpu().numpy() for emb in negative_embeddings]) if negative_embeddings else None
        self.noise_levels = np.array(noise_levels) if noise_levels else None
        
        # Calculate statistics from positive examples
        self.positive_centroid = np.mean(self.positive_embeddings, axis=0)
        self.positive_std = np.std(self.positive_embeddings, axis=0)
        
        # Calculate decision boundaries if negative samples exist
        if self.negative_embeddings is not None:
            self.negative_centroid = np.mean(self.negative_embeddings, axis=0)
            self._calculate_decision_boundary()
    
    def _calculate_decision_boundary(self):
        """Calculate optimal decision boundary using positive and negative samples"""
        pos_sims = self._batch_cosine_similarity(self.positive_embeddings, self.positive_centroid)
        neg_sims = self._batch_cosine_similarity(self.negative_embeddings, self.positive_centroid)
        
        # Find optimal threshold that maximizes separation
        all_sims = np.concatenate([pos_sims, neg_sims])
        all_sims.sort()
        
        best_threshold = 0
        best_separation = -float('inf')
        
        for threshold in all_sims:
            pos_correct = np.mean(pos_sims >= threshold)
            neg_correct = np.mean(neg_sims < threshold)
            separation = pos_correct + neg_correct - 1
            if separation > best_separation:
                best_separation = separation
                best_threshold = threshold
        
        self.decision_threshold = best_threshold
    
    def _batch_cosine_similarity(self, embeddings, reference):
        """Calculate cosine similarity between embeddings and reference"""
        # Ensure shapes are correct
        if len(embeddings.shape) == 3:
            embeddings = embeddings.squeeze(1)
        if len(reference.shape) == 1:
            reference = reference.reshape(1, -1)
            
        return cosine_similarity(embeddings, reference).flatten()
    
    def _adaptive_gaussian_kernel(self, distance, noise_level=0):
        """Compute Gaussian kernel with adaptive sigma based on noise level"""
        base_sigma = 0.4  # Increased to be more lenient with variations
        max_sigma = 0.6  # Increased accordingly
        
        adaptive_sigma = base_sigma + (max_sigma - base_sigma) * noise_level
        return norm.pdf(distance, loc=0, scale=adaptive_sigma)
    
    def compute_enhanced_similarity(self, query_embedding, noise_level=0):
        """Compute enhanced similarity score using multiple metrics"""
        # Ensure query_embedding is the right shape
        query_embedding = query_embedding.squeeze()
        
        # 1. Cosine similarity with positive centroid
        cosine_sim = cosine_similarity(
            query_embedding.reshape(1, -1), 
            self.positive_centroid.reshape(1, -1)
        )[0][0]
        
        # 2. Average similarity to positive examples
        pos_similarities = self._batch_cosine_similarity(self.positive_embeddings, query_embedding)
        avg_pos_sim = np.mean(pos_similarities)
        
        # 3. Distance from negative samples (if available)
        negative_penalty = 0
        if self.negative_embeddings is not None:
            neg_sims = self._batch_cosine_similarity(self.negative_embeddings, query_embedding)
            negative_penalty = np.mean(neg_sims)
        
        # 4. Gaussian kernel similarity with adaptive sigma
        embedding_distance = np.linalg.norm(query_embedding - self.positive_centroid)
        gaussian_sim = self._adaptive_gaussian_kernel(embedding_distance, noise_level)
        
        # 5. Standard deviation check (penalize outliers)
        std_penalty = np.mean(
            np.abs(query_embedding - self.positive_centroid) > 2 * self.positive_std
        )
        
        # Adjusted weights with more emphasis on positive similarities
        weights = {
            'cosine': 0.40,    # Increased to give more importance to direct similarity
            'avg_pos': 0.30,   # Increased to better handle new positive samples
            'gaussian': 0.20,  # Slightly reduced
            'negative': 0.15,  # Increased to better penalize negative samples
            'std': 0.05       # Kept same
        }
        
        # Adjust weights based on noise level
        if noise_level > 0.5:
            weights['gaussian'] += 0.05  # Reduced from 0.1
            weights['cosine'] -= 0.025   # Reduced penalty
            weights['avg_pos'] -= 0.025  # Reduced penalty
        
        final_score = (
            weights['cosine'] * cosine_sim +
            weights['avg_pos'] * avg_pos_sim +
            weights['gaussian'] * gaussian_sim -
            weights['negative'] * negative_penalty -
            weights['std'] * std_penalty
        )
        
        # Normalize score to [0, 1] range
        final_score = (final_score + 1) / 2
        
        return np.clip(final_score, 0, 1)  # Ensure score is between 0 and 1
    
    def is_wake_word(self, query_embedding, noise_level=0, threshold=None):
        """Determine if the query embedding represents the wake word"""
        similarity = self.compute_enhanced_similarity(query_embedding, noise_level)
        
        if threshold is None:
            # Change to
            # threshold = self.decision_threshold if hasattr(self, 'decision_threshold') else 0.55  # Lowered default threshold
            threshold = 0.55
            
        return similarity > threshold, similarity

def estimate_noise_level(audio, sr=16000):
    """Estimate noise level in audio signal"""
    signal_power = np.mean(audio ** 2)
    peak_power = np.max(audio ** 2)
    
    if peak_power > 0:
        snr = 10 * np.log10(peak_power / signal_power)
        noise_level = 1 / (1 + np.exp(0.1 * (snr - 10)))
    else:
        noise_level = 1.0
        
    return np.clip(noise_level, 0, 1)

if __name__ == "__main__":
    
    import librosa
    
    model = ONNXtoTorchModel(r"C:\Users\Rohit Francis\Desktop\Codes\Pathor Wake Word\EfficientWord-Net\eff_word_net\models\resnet_50_arc\slim_93%_accuracy_72.7390%.onnx")
    # audio = np.random.randn(24000)  # 1.5 seconds at 16kHz
    file_path = "audio2.wav"
    # Load and resample to 16 kHz
    audio, sr = librosa.load(file_path, sr=16000)
    # Ensure audio is exactly 24000 samples long
    expected_length = 24000
    if len(audio) < expected_length:
        pad_length = expected_length - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros

    print("Audio processing: ", audio.shape, sr)
    embeddings_audio = model(audio)
    print("Embeddings from audio shape:", embeddings_audio.shape)
    
    # audio = np.random.randn(24000)  # 1.5 seconds at 16kHz
    file_path2 = "audio2_twin.wav"
    # Load and resample to 16 kHz
    audio2, sr = librosa.load(file_path2, sr=16000)
    # Ensure audio is exactly 24000 samples long
    expected_length = 24000
    if len(audio2) < expected_length:
        pad_length = expected_length - len(audio2)
        audio2 = np.pad(audio2, (0, pad_length), mode='constant')  # Pad with zeros

    embeddings_audio2 = model(audio2)
         
    
    file_path = "audio.wav"
    # Load and resample to 16 kHz
    audio, sr = librosa.load(file_path, sr=16000)
    # Ensure audio is exactly 24000 samples long
    expected_length = 24000
    if len(audio) < expected_length:
        pad_length = expected_length - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros

    print("Audio processing: ", audio.shape, sr)
    embeddings_audio3 = model(audio)
    print("Embeddings from audio shape:", embeddings_audio3.shape)
    
    
    file_path = "audio_twin.wav"
    # Load and resample to 16 kHz
    audio, sr = librosa.load(file_path, sr=16000)
    # Ensure audio is exactly 24000 samples long
    expected_length = 24000
    if len(audio) < expected_length:
        pad_length = expected_length - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros

    print("Audio processing: ", audio.shape, sr)
    embeddings_audio4 = model(audio)
    print("Embeddings from audio shape:", embeddings_audio4.shape)
    
    positive_embeddings = [embeddings_audio, embeddings_audio2]
    negative_embeddings = [embeddings_audio3, embeddings_audio4]
    
    
    
    
    file_path = "test.wav"
    # Load and resample to 16 kHz
    audio, sr = librosa.load(file_path, sr=16000)
    # Ensure audio is exactly 24000 samples long
    expected_length = 24000
    if len(audio) < expected_length:
        pad_length = expected_length - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros

    query_embeddings = model(audio)
    query_embeddings = query_embeddings.detach().numpy()
    noise_levels = estimate_noise_level(audio)
    
    matcher = EnhancedSimilarityMatcher(positive_embeddings, negative_embeddings, noise_levels)    

    # For detection
    noise_level = estimate_noise_level(audio)
    is_wake_word, confidence = matcher.is_wake_word(query_embeddings, noise_level)

    print(is_wake_word, confidence)



# Use Below command to test the model
# ffmpeg -i "C:\Users\Rohit Francis\Desktop\Codes\Pathor Wake Word\MLCOMMONWORDS_EN\mswc_microset\en\clips\bed\common_voice_en_11350.opus"  test.wav





# import librosa
# import numpy as np

# file_path = "audio.wav"
# # Load and resample to 16 kHz
# audio, sr = librosa.load(file_path, sr=16000)
# print(f"Before {audio.shape}")
# # Ensure audio is exactly 24000 samples long
# expected_length = 24000
# if len(audio) < expected_length:
#     pad_length = expected_length - len(audio)
#     audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros

# print(audio[17000:17005])
# print("Processed Audio Shape:", audio.shape)  # Should be (24000,)


# import librosa

# file_path = "audio.wav"
# audio, sr = librosa.load(file_path, sr=24000)  # Converts to mono and resamples to 16kHz
# print(audio.shape, sr)

# import tensorflow.lite as tflite
# import os

# logmelcalc_interpreter = tflite.Interpreter(
#                 model_path=r"C:\Users\Rohit Francis\Desktop\Codes\Pathor Wake Word\EfficientWord-Net\eff_word_net\models\first_iteration_siamese\logmelcalc.tflite"
#         )
# logmelcalc_interpreter.allocate_tensors()

# import torch
# import torch.nn as nn
# import onnxruntime
# import numpy as np
# from torchvision.models import resnet50

# class ONNXtoTorchModel(nn.Module):
#     def __init__(self, onnx_path):
#         super().__init__()
#         # Load ONNX model
#         self.session = onnxruntime.InferenceSession(onnx_path)
#         self.input_name = self.session.get_inputs()[0].name
#         self.input_shape = self.session.get_inputs()[0].shape
        
#     def forward(self, x):
#         # Convert PyTorch tensor to numpy for ONNX Runtime
#         if isinstance(x, torch.Tensor):
#             x = x.detach().cpu().numpy()
            
#         # Run inference
#         outputs = self.session.run(None, {self.input_name: x})
#         out = torch.from_numpy(outputs[0])
        
#         # Process the out further to get the embeddings which will be used with AdaptiveSimilarity

