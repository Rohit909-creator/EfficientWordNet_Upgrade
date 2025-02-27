import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import onnxruntime
import numpy as np
from python_speech_features import logfbank
import librosa
from UTILS import enhance_similarity_scores, preprocess_embeddings
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import norm
from colorama import Fore, Style


class ONNXtoTorchModel(nn.Module):
    def __init__(self, onnx_path):
        super().__init__()
        # Load ONNX model
        self.session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.window_length = 1.5
        self.window_frames = int(self.window_length * 16000)
        
    def compute_logfbank_features(self, inpAudio):
        """
        Compute log Mel-filterbank features
        """
        return logfbank(
            inpAudio,
            samplerate=16000,
            winlen=0.025,
            winstep=0.01,
            nfilt=64,
            nfft=512,
            preemph=0.0
        )
    
    def get_embeddings(self, audio):
        """
        Convert audio to embeddings
        Args:
            audio: numpy array of shape (window_frames,) - 1.5 seconds of audio at 16kHz
        Returns:
            embeddings: torch tensor of embeddings
        """
        assert audio.shape == (self.window_frames,), f"Expected audio shape {self.window_frames}, got {audio.shape}"
        
        # Compute log mel features
        features = self.compute_logfbank_features(audio)
        
        # Add batch and channel dimensions
        features = np.expand_dims(features, axis=(0,1))
        features = np.float32(features)
        
        # Get embeddings from ONNX model
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: features}
        )[0]
        
        return torch.from_numpy(outputs)
    
    def forward(self, x):
        """
        Forward pass - handles both audio and pre-computed mel spectrograms
        Args:
            x: Either audio waveform or mel spectrogram
        Returns:
            embeddings: torch tensor of embeddings
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            
        if len(x.shape) == 1:  # Audio waveform
            return self.get_embeddings(x)
        else:  # Mel spectrogram
            # Add batch and channel dimensions if needed
            if len(x.shape) == 2:
                x = np.expand_dims(x, axis=(0,1))
            x = np.float32(x)
            outputs = self.session.run([self.output_name], {self.input_name: x})[0]
            return torch.from_numpy(outputs)
        
    def compute_similarity(self, emb1, emb2):
        """
        Compute similarity between two embeddings using cosine similarity
        """
        if isinstance(emb1, torch.Tensor):
            emb1 = emb1.detach().cpu().numpy()
        if isinstance(emb2, torch.Tensor):
            emb2 = emb2.detach().cpu().numpy()
            
        cosine_similarity = np.matmul(emb2, emb1.T)
        confidence_score = (cosine_similarity + 1) / 2
        return confidence_score.max()

    def enhanced_similarity(self, emb1, emb2, test=False):
        
        embs1_processed = preprocess_embeddings(emb1)
        embs2_processed = preprocess_embeddings(emb2)
        if test:
            print("Enhanced Similarity Cosine:", enhance_similarity_scores(embs1_processed, embs2_processed))
            print("Enhanced Similarity Gaussian:", enhance_similarity_scores(embs1_processed, embs2_processed, method="gaussian"))
            print("Enhanced Similarity Angular:", enhance_similarity_scores(embs1_processed, embs2_processed, method="angular"))
            print("Enhanced Similarity Combined:", enhance_similarity_scores(embs1_processed, embs2_processed, method="combined"))

        if not test:
            cosine_sim = enhance_similarity_scores(embs1_processed, embs2_processed)
            gausian_sim = enhance_similarity_scores(embs1_processed, embs2_processed, method="gaussian")
            angular_sim = enhance_similarity_scores(embs1_processed, embs2_processed, method="angular")
            combined_sim = enhance_similarity_scores(embs1_processed, embs2_processed, method="combined")
        
        
        return cosine_sim, gausian_sim, angular_sim, combined_sim


def make_reference(model, name, audio):
    
    import json
    embeddings = model(audio)
    
    d = {"name": name, "embeddings": embeddings.tolist()}
    
    with open("path_to_reference.json", 'w') as f:
        s = json.dumps(d)
        f.write(s)


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
            'cosine': 0.40,      # Keep this as is - it's a strong indicator
            'avg_pos': 0.30,     # Keep this as is - it's helpful for faint voices
            'gaussian': 0.15,    # Keep this as is, The Gaussian gives more credit to samples that are "in the neighborhood" of your centroid.
            'negative': 0.40,    # Increase from 0.10 to 0.20 - put more weight on negative examples
            'std': 0.10          # Increase from 0.03 to 0.05 - be more strict about standard deviation
        }
        
        # OLD WEIGHTS based on test results analysis
        # weights = {
        #     'cosine': 0.45,      # Keep this as is - it's a strong indicator
        #     'avg_pos': 0.35,     # Keep this as is - it's helpful for faint voices
        #     'gaussian': 0.15,    # Keep this as is, The Gaussian gives more credit to samples that are "in the neighborhood" of your centroid.
        #     'negative': 0.20,    # Increase from 0.10 to 0.20 - put more weight on negative examples
        #     'std': 0.05          # Increase from 0.03 to 0.05 - be more strict about standard deviation
        # }

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
            threshold = 0.67
            # print(f"Decision Threshold set to {threshold} based on current noise_levels")
            # threshold = 0.60
            
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
    

# Example usage:
if __name__ == "__main__":
    model = ONNXtoTorchModel("./resnet_50_arc/slim_93%_accuracy_72.7390%.onnx")
    
    # file_path = "audio2.wav"
    # # Load and resample to 16 kHz
    # audio, sr = librosa.load(file_path, sr=16000)
    # # Ensure audio is exactly 24000 samples long
    # expected_length = 24000
    # if len(audio) < expected_length:
    #     pad_length = expected_length - len(audio)
    #     audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros

    # print("Audio processing: ", audio.shape, sr)
    
    # make_reference(model, "bed", audio)
    
    # # Test with audio
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
    file_path2 = "audio_twin.wav"
    # Load and resample to 16 kHz
    audio2, sr = librosa.load(file_path2, sr=16000)
    # Ensure audio is exactly 24000 samples long
    expected_length = 24000
    if len(audio2) < expected_length:
        pad_length = expected_length - len(audio2)
        audio2 = np.pad(audio2, (0, pad_length), mode='constant')  # Pad with zeros

    print("Audio processing: ", audio2.shape, sr)
    embeddings_audio2 = model(audio2)
    print("Embeddings from audio2 shape:", embeddings_audio2.shape)
    
    # Test with mel spectrogram
    # mel_spec = np.random.randn(149, 64)  # Example mel spectrogram shape
    # embeddings_mel = model(mel_spec)
    # print("Embeddings from mel spectrogram shape:", embeddings_mel.shape)
    
    # Test similarity computation
    
    print(f"Trial with audio files {file_path} and {file_path2}")
    similarity = model.compute_similarity(embeddings_audio, embeddings_audio2)
    print("Old Cosine Similarity score:", similarity)
    
    model.enhanced_similarity(embeddings_audio, embeddings_audio2)
