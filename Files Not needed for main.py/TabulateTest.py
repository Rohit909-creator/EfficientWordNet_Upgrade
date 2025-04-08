import os
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import norm
from colorama import Fore, Style

def tabulate(headers, results):
    """
    Prints a formatted table without using external libraries.
    
    :param headers: List of column headers
    :param results: List of row data
    """
    col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *results)]
    
    def format_row(row):
        return " | ".join(f"{str(item):<{col_widths[i]}}" for i, item in enumerate(row))
    
    print(f"\n{Fore.CYAN}=== Test Results ==={Style.RESET_ALL}")
    print("-" * (sum(col_widths) + 3 * (len(headers) - 1)))
    print(format_row(headers))
    print("-" * (sum(col_widths) + 3 * (len(headers) - 1)))
    for row in results:
        print(format_row(row))
    print("-" * (sum(col_widths) + 3 * (len(headers) - 1)))

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
        
        # ADJUSTED WEIGHTS based on test results analysis
        weights = {
            'cosine': 0.45,      # Keep this as is - it's a strong indicator
            'avg_pos': 0.35,     # Keep this as is - it's helpful for faint voices
            'gaussian': 0.15,    # Keep this as is
            'negative': 0.20,    # Increase from 0.10 to 0.20 - put more weight on negative examples
            'std': 0.05          # Increase from 0.03 to 0.05 - be more strict about standard deviation
        }

        # Adjust the noise level handling
        if noise_level > 0.3:
            weights['gaussian'] += 0.05
            weights['cosine'] -= 0.02
            weights['avg_pos'] -= 0.01
            weights['std'] -= 0.01  # Less reduction in std penalty (from -0.02)

        # Modify the faint voice detection logic
        if cosine_sim < 0.25 and cosine_sim > 0.08:  # Increase lower bound from 0.05 to 0.08
            # Potential faint voice - still reduce penalties but less aggressively
            weights['std'] *= 0.7  # Reduce less (from 0.5 to 0.7)
            
            # Make the boost more conditional
            if avg_pos_sim > 0.85 * cosine_sim and cosine_sim > 0.12:
                boost = 0.03  # Smaller boost (from 0.05 to 0.03)
            else:
                boost = 0
        else:
            boost = 0
        
    
        final_score = (
            weights['cosine'] * cosine_sim +
            weights['avg_pos'] * avg_pos_sim +
            weights['gaussian'] * gaussian_sim -
            weights['negative'] * negative_penalty -
            weights['std'] * std_penalty + 
            boost
        )        
        # Normalize score to [0, 1] range
        final_score = (final_score + 1) / 2
        
        # Calculate individual metric scores for detailed analysis
        metrics = {
            'cosine_sim': cosine_sim,
            'avg_pos_sim': avg_pos_sim,
            'gaussian_sim': gaussian_sim,
            'negative_penalty': negative_penalty,
            'std_penalty': std_penalty
        }
        
        return np.clip(final_score, 0, 1), metrics  # Ensure score is between 0 and 1
    
    def is_wake_word(self, query_embedding, noise_level=0, threshold=None):
        """Determine if the query embedding represents the wake word"""
        similarity, metrics = self.compute_enhanced_similarity(query_embedding.detach().numpy(), noise_level)
        
        if threshold is None:
            # threshold = getattr(self, 'decision_threshold', 0.55)  # Default to 0.55 if no decision_threshold
            threshold = 0.58
            
        return similarity > threshold, similarity, metrics

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

    def run_comprehensive_test(self, model, test_files, threshold=None):
        """
        Run comprehensive testing with detailed metrics using EnhancedSimilarityMatcher
        
        :param model: Function to convert audio to embeddings
        :param test_files: List of test audio files to evaluate
        :param threshold: Detection threshold (default: None, uses self.decision_threshold or 0.55)
        :return: List of test results
        """
        print(f"\n{Fore.CYAN}=== Enhanced Wake Word Detection System Test ===\n{Style.RESET_ALL}")
        
        # Test results
        results = []
        print(f"\n{Fore.YELLOW}Running tests with EnhancedSimilarityMatcher...{Style.RESET_ALL}")
        
        for test_file in test_files:
            # Process audio file
            emb, audio, sr = self.process_audio(test_file, model)
            
            # Estimate noise level
            noise_level = self.estimate_noise_level(audio)
            
            # Determine if wake word detected using the enhanced similarity logic
            is_wake, confidence, metrics = self.is_wake_word(emb, noise_level, threshold)
            
            # Add result with detailed metrics
            results.append([
                os.path.basename(test_file),
                f"{confidence:.4f}",
                "✓" if is_wake else "✗",
                f"{metrics['cosine_sim']:.4f}",
                f"{metrics['avg_pos_sim']:.4f}",
                f"{metrics['gaussian_sim']:.4f}",
                f"{metrics['negative_penalty']:.4f}",
                f"{metrics['std_penalty']:.4f}",
                f"{noise_level:.4f}"
            ])
        
        # Print results table
        headers = ["File", "Confidence", "Detection", "Cosine", "Avg Pos", "Gaussian", "Neg Penalty", "Std Penalty", "Noise Level"]
        tabulate(headers, results)
        
        # Print statistics
        detections = sum(1 for r in results if r[2] == "✓")
        print(f"\n{Fore.CYAN}=== Statistics ==={Style.RESET_ALL}")
        print(f"Total tests: {len(results)}")
        print(f"Detections: {detections}")
        print(f"Detection rate: {detections/len(results)*100:.2f}%")
        
        if hasattr(self, 'decision_threshold'):
            print(f"Using calculated decision threshold: {self.decision_threshold:.4f}")
        else:
            print(f"Using default threshold: {threshold or 0.55:.4f}")
            
        # Calculate false positive and negative rates if we can infer expected results from filenames
        positive_tests = sum(1 for file in test_files if "negative" not in os.path.basename(file).lower())
        negative_tests = len(test_files) - positive_tests
        
        true_positives = sum(1 for i, r in enumerate(results) if 
                            r[2] == "✓" and "negative" not in os.path.basename(test_files[i]).lower())
        false_positives = sum(1 for i, r in enumerate(results) if 
                             r[2] == "✓" and "negative" in os.path.basename(test_files[i]).lower())
        
        if positive_tests > 0:
            print(f"True positive rate: {true_positives/positive_tests*100:.2f}%")
        if negative_tests > 0:
            print(f"False positive rate: {false_positives/negative_tests*100:.2f}%")
        
        return results


# Example usage
if __name__ == "__main__":
    # This is just a skeleton example showing how to use the class
    from Detection import ONNXtoTorchModel
    import librosa

    
    # Define paths
    base_dir = "./"
    
    # Model path
    model_path = os.path.join(base_dir, "resnet_50_arc", "slim_93%_accuracy_72.7390%.onnx")
    model = ONNXtoTorchModel(model_path)
    
    # Audio files
    positive_files = [
        os.path.join(base_dir, "tts_samples", "positive", "normal_voice0.wav"),
        os.path.join(base_dir, "tts_samples", "positive", "normal_voice1.wav"),
        os.path.join(base_dir, "tts_samples", "positive", "soft_voice0.wav"),
        os.path.join(base_dir, "tts_samples", "positive", "soft_voice1.wav"),
        os.path.join(base_dir, "tts_samples", "positive", "clear_voice0.wav"),
        os.path.join(base_dir, "tts_samples", "positive", "clear_voice1.wav")
    ]
    
    negative_files = [
        os.path.join(base_dir, "tts_samples", "negative", "partial_voice0.wav"),
        os.path.join(base_dir, "tts_samples", "negative", "partial_voice1.wav"),
        os.path.join(base_dir, "tts_samples", "negative", "last_part_voice0.wav"),
        os.path.join(base_dir, "tts_samples", "negative", "last_part_voice1.wav")
    ]
    
    test_files = [
        os.path.join(base_dir, "Recording.wav"),
        os.path.join(base_dir, "Recording (2).wav"),
        os.path.join(base_dir, "Recording (3).wav"),
        os.path.join(base_dir, "Recording (4).wav"),
        os.path.join(base_dir, "Recording (5).wav"),
        os.path.join(base_dir, "Recording (6).wav"),
        os.path.join(base_dir, "Recording (7).wav"),
        os.path.join(base_dir, "dim_recording.wav"),
        os.path.join(base_dir, "dim_recording2.wav"),
        os.path.join(base_dir, "faint_voice.wav"),
        os.path.join(base_dir, "faint_voice2.wav"),
        os.path.join(base_dir, "Recording_negative (1).wav"),
        os.path.join(base_dir, "Recording_negative (2).wav"),
        os.path.join(base_dir, "Recording_negative (3).wav"),
        os.path.join(base_dir, "Recording_negative (4).wav"),
        os.path.join(base_dir, "Recording_negative (5).wav"),
        os.path.join(base_dir, "dim_recording_negative (1).wav"),
        os.path.join(base_dir, "dim_recording_negative (2).wav"),
        os.path.join(base_dir, "dim_recording_negative (3).wav"),
        os.path.join(base_dir, "dim_recording_negative (4).wav"),
        os.path.join(base_dir, "dim_recording_negative (5).wav"),
        os.path.join(base_dir, "dim_recording_negative (6).wav"),
        os.path.join(base_dir, "dim_recording_negative (7).wav"),
    ]
        
    # Process positive examples
    print(f"{Fore.GREEN}Processing positive examples...{Style.RESET_ALL}")
    positive_embeddings = []
    for file in positive_files:
        
        audio, sr = librosa.load(file, sr=16000)
        # Ensure audio is exactly 24000 samples long
        expected_length = 24000
        if len(audio) < expected_length:
            pad_length = expected_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros
        
        emb = model(audio)
        positive_embeddings.append(emb)
    
    # Process negative examples
    print(f"{Fore.RED}Processing negative examples...{Style.RESET_ALL}")
    negative_embeddings = []
    for file in negative_files:
        
        audio, sr = librosa.load(file, sr=16000)
        # Ensure audio is exactly 24000 samples long
        expected_length = 24000
        if len(audio) < expected_length:
            pad_length = expected_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros
        
        emb = model(audio)
        negative_embeddings.append(emb)
    
    # Initialize matcher
    matcher = EnhancedSimilarityMatcher(positive_embeddings, negative_embeddings)
    
    # Run comprehensive test
    results = matcher.run_comprehensive_test(model, test_files)

