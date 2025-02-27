# adaptive_tester.py

import os
import numpy as np
import librosa
import time
import json
import pyaudio
import queue
import threading
from Detection import ONNXtoTorchModel
from colorama import Fore, Style

class AdaptiveSimilarityMatcher:
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
        
        # Initialize weights with default values
        self.weights = {
            'cosine': 0.40,
            'avg_pos': 0.30,
            'gaussian': 0.15,
            'negative': 0.40,
            'std': 0.10
        }
        
        # History for tracking changes
        self.history = {
            'weights': [self.weights.copy()],
            'results': []
        }
        
        # Calculate decision boundaries if negative samples exist
        if self.negative_embeddings is not None:
            self.negative_centroid = np.mean(self.negative_embeddings, axis=0)
            self._calculate_decision_boundary()
    
    def _calculate_decision_boundary(self):
        """Calculate optimal decision boundary using positive and negative samples"""
        from sklearn.metrics.pairwise import cosine_similarity
        
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
        print(f"Calculated optimal decision threshold: {self.decision_threshold:.4f}")
    
    def _batch_cosine_similarity(self, embeddings, reference):
        """Calculate cosine similarity between embeddings and reference"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Ensure shapes are correct
        if len(embeddings.shape) == 3:
            embeddings = embeddings.squeeze(1)
        if len(reference.shape) == 1:
            reference = reference.reshape(1, -1)
            
        return cosine_similarity(embeddings, reference).flatten()
    
    def _adaptive_gaussian_kernel(self, distance, noise_level=0):
        """Compute Gaussian kernel with adaptive sigma based on noise level"""
        from scipy.stats import norm
        
        base_sigma = 0.4  # Increased to be more lenient with variations
        max_sigma = 0.6   # Increased accordingly
        
        adaptive_sigma = base_sigma + (max_sigma - base_sigma) * noise_level
        return norm.pdf(distance, loc=0, scale=adaptive_sigma)
    
    def compute_enhanced_similarity(self, query_embedding, noise_level=0):
        """Compute enhanced similarity score using multiple metrics"""
        from sklearn.metrics.pairwise import cosine_similarity
        
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
        
        # 3. Time-aligned similarity for beginning of word
        # (This is a simplified approximation since we don't have direct time info in embeddings)
        # We'll simulate it by giving more weight to certain dimensions
        first_third_sim = cosine_similarity(
            query_embedding[:query_embedding.shape[0]//3].reshape(1, -1),
            self.positive_centroid[:self.positive_centroid.shape[0]//3].reshape(1, -1)
        )[0][0]
        
        # 4. Distance from negative samples (if available)
        negative_penalty = 0
        if self.negative_embeddings is not None:
            neg_sims = self._batch_cosine_similarity(self.negative_embeddings, query_embedding)
            negative_penalty = np.mean(neg_sims)
        
        # 5. Gaussian kernel similarity with adaptive sigma
        embedding_distance = np.linalg.norm(query_embedding - self.positive_centroid)
        gaussian_sim = self._adaptive_gaussian_kernel(embedding_distance, noise_level)
        
        # 6. Standard deviation check (penalize outliers)
        std_penalty = np.mean(
            np.abs(query_embedding - self.positive_centroid) > 2 * self.positive_std
        )
                
        # Apply weights to each component
        final_score = (
            self.weights['cosine'] * cosine_sim +
            self.weights['avg_pos'] * avg_pos_sim +
            self.weights['gaussian'] * gaussian_sim -
            self.weights['negative'] * negative_penalty -
            self.weights['std'] * std_penalty +
            # Give extra weight to the beginning of the word for "A" in "Alexa"
            0.15 * first_third_sim
        )
        
        # Normalize score to [0, 1] range
        final_score = (final_score + 1) / 2
        
        # Calculate individual metric scores for detailed analysis
        metrics = {
            'cosine_sim': cosine_sim,
            'avg_pos_sim': avg_pos_sim,
            'first_third_sim': first_third_sim,
            'gaussian_sim': gaussian_sim,
            'negative_penalty': negative_penalty,
            'std_penalty': std_penalty
        }
        
        return np.clip(final_score, 0, 1), metrics
    
    def is_wake_word(self, query_embedding, noise_level=0, threshold=None):
        """Determine if the query embedding represents the wake word"""
        if hasattr(query_embedding, 'detach'):
            query_embedding = query_embedding.detach().cpu().numpy()
        similarity, metrics = self.compute_enhanced_similarity(query_embedding, noise_level)
        
        if threshold is None:
            threshold = getattr(self, 'decision_threshold', 0.60)
            
        return similarity > threshold, similarity, metrics
    
    def set_weights(self, weights):
        """Set weights for similarity computation"""
        self.weights = weights.copy()
        self.history['weights'].append(self.weights.copy())
    
    def adjust_weights(self, should_detect, metrics):
        """
        Adjust weights based on feedback
        
        Args:
            should_detect: True if should have detected, False otherwise
            metrics: Dictionary of similarity metrics from last detection
        """
        adjustment = 0.05  # Small adjustment step
        
        if should_detect:  # Should have detected - increase weights for strong indicators
            # Find the strongest positive indicator
            strongest = max(metrics['cosine_sim'], metrics['avg_pos_sim'], metrics['gaussian_sim'])
            
            if metrics['cosine_sim'] == strongest:
                self.weights['cosine'] += adjustment
            elif metrics['avg_pos_sim'] == strongest:
                self.weights['avg_pos'] += adjustment
            elif metrics['gaussian_sim'] == strongest:
                self.weights['gaussian'] += adjustment
                
            # Decrease negative weights
            self.weights['negative'] -= adjustment/2
            self.weights['std'] -= adjustment/2
                
        else:  # Shouldn't have detected - decrease weights for misleading indicators
            # Find the strongest positive indicator
            strongest = max(metrics['cosine_sim'], metrics['avg_pos_sim'], metrics['gaussian_sim'])
            
            if metrics['cosine_sim'] == strongest:
                self.weights['cosine'] -= adjustment
            elif metrics['avg_pos_sim'] == strongest:
                self.weights['avg_pos'] -= adjustment
            elif metrics['gaussian_sim'] == strongest:
                self.weights['gaussian'] -= adjustment
                
            # Increase negative weights
            self.weights['negative'] += adjustment/2
            self.weights['std'] += adjustment/2
        
        # Normalize weights to prevent extreme values
        self._normalize_weights()
        
        # Store updated weights in history
        self.history['weights'].append(self.weights.copy())
    
    def _normalize_weights(self):
        """Ensure weights remain in reasonable ranges"""
        for key in self.weights:
            self.weights[key] = max(0.05, min(0.6, self.weights[key]))
    
    def save_weights(self, filename):
        """Save the current weights to a file"""
        with open(filename, 'w') as f:
            json.dump(self.weights, f, indent=4)
        print(f"Weights saved to {filename}")
    
    def load_weights(self, filename):
        """Load weights from a file"""
        try:
            with open(filename, 'r') as f:
                self.weights = json.load(f)
            print(f"Weights loaded from {filename}")
            self.history['weights'].append(self.weights.copy())
        except FileNotFoundError:
            print(f"Weights file {filename} not found, using defaults")
    
    def process_audio(self, file_path, model, expected_length=24000):
        """Process audio file and return embeddings"""
        print(f"Processing: {os.path.basename(file_path)}")
        audio, sr = librosa.load(file_path, sr=16000)
        if len(audio) < expected_length:
            audio = np.pad(audio, (0, expected_length - len(audio)), mode='constant')
        audio = audio[:expected_length]
        return model(audio), audio, sr
    
    def estimate_noise_level(self, audio):
        """Estimate noise level in audio signal"""
        signal_power = np.mean(audio ** 2)
        peak_power = np.max(audio ** 2)
        
        if peak_power > 0:
            snr = 10 * np.log10(peak_power / signal_power)
            noise_level = 1 / (1 + np.exp(0.1 * (snr - 10)))
        else:
            noise_level = 1.0
        return np.clip(noise_level, 0, 1)
    
    def visualize_improvement(self, output_dir="./"):
        """
        Create visualization to show improvement in detection
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot weights before and after
            plt.figure(figsize=(15, 10))
            
            # First subplot: Initial vs Final weights
            plt.subplot(2, 2, 1)
            initial_weights = self.history['weights'][0]
            final_weights = self.history['weights'][-1]
            
            x = np.arange(len(initial_weights))
            width = 0.35
            
            plt.bar(x - width/2, initial_weights.values(), width, label='Initial')
            plt.bar(x + width/2, final_weights.values(), width, label='Final')
            
            plt.xlabel('Weight Parameter')
            plt.ylabel('Weight Value')
            plt.title('Initial vs Final Weights')
            plt.xticks(x, initial_weights.keys(), rotation=45)
            plt.ylim(0, 0.7)
            plt.legend()
            
            # Second subplot: Weight evolution over time
            plt.subplot(2, 2, 2)
            weight_history = np.array([[w[key] for key in initial_weights.keys()] for w in self.history['weights']])
            
            for i, key in enumerate(initial_weights.keys()):
                plt.plot(weight_history[:, i], label=key)
            
            plt.xlabel('Adjustment Iteration')
            plt.ylabel('Weight Value')
            plt.title('Weight Evolution')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add accuracy plot if results are available
            if 'accuracy' in self.history:
                plt.subplot(2, 2, 3)
                plt.plot(self.history['accuracy'], marker='o')
                plt.xlabel('Test Iteration')
                plt.ylabel('Accuracy')
                plt.title('Accuracy Improvement')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            
            # Add confusion statistics if available
            if 'true_positives' in self.history and 'false_positives' in self.history:
                plt.subplot(2, 2, 4)
                plt.plot(self.history['true_positives'], marker='o', label='True Positives')
                plt.plot(self.history['false_positives'], marker='x', label='False Positives')
                plt.xlabel('Test Iteration')
                plt.ylabel('Rate')
                plt.title('Detection Performance')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'weight_improvement.png'))
            print(f"Visualization saved to {os.path.join(output_dir, 'weight_improvement.png')}")
            
            # Close the plot to free memory
            plt.close()
            
            return True
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            return False


class SimpleMicStream:
    """Handles real-time audio capture from microphone"""
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_running = False
        
    def start_stream(self):
        """Start capturing audio from microphone"""
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        self.is_running = True
        print(f"{Fore.GREEN}Microphone stream started{Style.RESET_ALL}")
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def getFrame(self):
        """Get latest audio frame from queue"""
        if not self.is_running:
            return None
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
        
    def stop_stream(self):
        """Stop and clean up audio stream"""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print(f"{Fore.RED}Microphone stream stopped{Style.RESET_ALL}")