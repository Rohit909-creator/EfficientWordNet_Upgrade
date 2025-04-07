import tensorflow as tf
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
from colorama import Fore, Style

def cosine_similarity(a, b, axis=1, eps=1e-8):
    """
    Compute cosine similarity between tensors a and b along specified axis
    Handles different input dimensions appropriately
    """
    # Check dimensions and adjust axis if needed
    if len(tf.shape(a)) == 1:
        a = tf.expand_dims(a, 0)
        if axis == 1:
            axis = 0
    
    if len(tf.shape(b)) == 1:
        b = tf.expand_dims(b, 0)
        if axis == 1:
            axis = 0
    
    # Normalize and compute similarity
    a_norm = tf.nn.l2_normalize(a, axis=axis, epsilon=eps)
    b_norm = tf.nn.l2_normalize(b, axis=axis, epsilon=eps)
    
    # Ensure output has appropriate dimensions
    return tf.reduce_sum(a_norm * b_norm, axis=axis)


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


class EnhancedSimilarityMatcher(tf.keras.Model):
    
    def __init__(self, positive_embeddings, negative_embeddings=None, noise_levels=None):
        super().__init__()
        
        self.positive_embeddings:tf.Tensor = tf.constant([emb.squeeze().cpu().numpy() for emb in positive_embeddings])
        self.negative_embeddings:tf.Tensor = tf.constant([emb.squeeze().cpu().numpy() for emb in negative_embeddings])
        self.noise_levels = tf.constant(noise_levels) if noise_levels else None
        
        # Calculate statistics from positive examples
        self.positive_centroid = tf.reduce_mean(self.positive_embeddings,axis=0)
        self.positive_std = tf.math.reduce_std(self.positive_embeddings, axis=0)
        
        if self.negative_embeddings is not None:
            self.negative_embeddings = tf.reduce_mean(self.negative_embeddings, axis=0)
            self._calculate_decision_boundary()
            
            
    def call(self, query_embedding, noise_level):
        check, similarity, metrics = self.is_wake_word(query_embedding, noise_level)
        return check, similarity, metrics
    
    def _calculate_decision_boundary(self):
        """Calculate optimal decision threshold based on positive and negative examples"""
        # Get similarity scores for positive examples
        pos_sims = self._batch_cosine_similarity(self.positive_embeddings, self.positive_centroid)
        
        # Properly handle negative embeddings
        if hasattr(self, 'negative_embeddings') and self.negative_embeddings is not None:
            if isinstance(self.negative_embeddings, tf.Tensor) and len(tf.shape(self.negative_embeddings)) > 1:
                neg_sims = self._batch_cosine_similarity(self.negative_embeddings, self.positive_centroid)
            else:
                # Handle case where negative_embeddings is a single vector
                neg_sims = cosine_similarity(self.negative_embeddings, self.positive_centroid)
                neg_sims = tf.reshape(neg_sims, [-1])
        else:
            neg_sims = tf.constant([], dtype=tf.float32)
        
        # Combine and sort similarities
        all_sims = tf.concat([pos_sims, neg_sims], axis=0)
        all_sims = tf.sort(all_sims)
        
        # Find best threshold
        best_threshold = tf.constant(0.0, dtype=tf.float32)
        best_separation = tf.constant(-float('inf'), dtype=tf.float32)
        
        # Use TensorFlow while loop for threshold search
        def condition(i, best_threshold, best_separation):
            return i < tf.shape(all_sims)[0]
        
        def body(i, best_threshold, best_separation):
            threshold = all_sims[i]
            pos_correct = tf.reduce_mean(tf.cast(pos_sims >= threshold, tf.float32))
            neg_correct = tf.reduce_mean(tf.cast(neg_sims < threshold, tf.float32))
            separation = pos_correct + neg_correct - 1.0
            
            new_best_threshold = tf.cond(
                separation > best_separation,
                lambda: threshold,
                lambda: best_threshold
            )
            
            new_best_separation = tf.maximum(separation, best_separation)
            
            return i + 1, new_best_threshold, new_best_separation
        
        # Initial values
        i = tf.constant(0)
        
        # Run the loop
        _, threshold, _ = tf.while_loop(
            condition, 
            body, 
            [i, best_threshold, best_separation]
        )
        
        self.decision_threshold = threshold
        
    def _batch_cosine_similarity(self, embeddings:tf.Tensor, reference:tf.Tensor):
        """
        Compute cosine similarity between batches of embeddings and a reference
        Handles different input dimensions appropriately
        """
        # Handle potential 3D inputs
        if len(tf.shape(embeddings)) == 3:
            embeddings = tf.squeeze(embeddings)
        
        # Ensure reference has proper shape
        if len(tf.shape(reference)) == 1:
            reference = tf.reshape(reference, [1, -1])
        
        # Handle case where embeddings is a single vector (not a batch)
        if len(tf.shape(embeddings)) == 1:
            embeddings = tf.expand_dims(embeddings, 0)
            sim = cosine_similarity(embeddings, reference, axis=0)
            return sim
        
        # Normal batch case
        sim = cosine_similarity(embeddings, reference, axis=1)
        
        # Reshape if needed
        if len(tf.shape(sim)) > 1:
            output = tf.reshape(sim, [tf.shape(sim)[0], -1])
        else:
            output = sim
            
        return output
    
    def _adaptive_gaussian_kernel_tf(self, distance, noise_level=0.0):
        base_sigma = 0.4
        max_sigma = 0.6
        adaptive_sigma = base_sigma + (max_sigma - base_sigma) * noise_level
        # Gaussian PDF
        pi = tf.constant(3.14159265358979323846)
        coeff = 1.0 / (adaptive_sigma * tf.sqrt(2.0 * pi))
        exponent = -0.5 * tf.square(distance / adaptive_sigma)
        return coeff * tf.exp(exponent)

    def compute_enhanced_similarity(self, query_embedding, noise_level=0):
        """Compute enhanced similarity score using multiple metrics"""
        
        # Ensure query_embedding is the right shape
        query_embedding = tf.squeeze(query_embedding)
        
        # 1. Cosine similarity with positive centroid
        cosine_sim = cosine_similarity(
            tf.reshape(query_embedding, [1, -1]), 
            tf.reshape(self.positive_centroid, [1, -1])
        )[0]
        
        # 2. Average similarity to positive examples
        pos_similarities = self._batch_cosine_similarity(self.positive_embeddings, query_embedding)
        avg_pos_sim = tf.reduce_mean(pos_similarities)
        
        # 3. Distance from negative samples (if available)
        negative_penalty = tf.constant(0.0)
        if self.negative_embeddings is not None:
            neg_sims = self._batch_cosine_similarity(self.negative_embeddings, query_embedding)
            negative_penalty = tf.reduce_mean(neg_sims)
        
        # 4. Gaussian kernel similarity with adaptive sigma
        embedding_distance = tf.norm(query_embedding - self.positive_centroid)
        gaussian_sim = self._adaptive_gaussian_kernel_tf(embedding_distance, noise_level)
        
        # 5. Standard deviation check (penalize outliers)
        std_penalty = tf.reduce_mean(
            tf.cast(tf.abs(query_embedding - self.positive_centroid) > 2 * self.positive_std, tf.float32)
        )
        
        # Define weights
        weights = {
            'cosine': tf.constant(0.45, dtype=tf.float32),
            'avg_pos': tf.constant(0.35, dtype=tf.float32),
            'gaussian': tf.constant(0.15, dtype=tf.float32),
            'negative': tf.constant(0.20, dtype=tf.float32),
            'std': tf.constant(0.05, dtype=tf.float32)
        }

        # Adjust the noise level handling
        # noise_level_tensor = tf.constant(noise_level, dtype=tf.float32)
        # Don't convert noise_level to a tensor if it's already a tensor
        if not isinstance(noise_level, tf.Tensor):
            noise_level_tensor = tf.constant(noise_level, dtype=tf.float32)
        else:
            noise_level_tensor = noise_level
        
        noise_level_condition = tf.greater(noise_level_tensor, 0.3)
        
        gaussian_weight = tf.cond(
            noise_level_condition,
            lambda: weights['gaussian'] + 0.05,
            lambda: weights['gaussian']
        )
        
        cosine_weight = tf.cond(
            noise_level_condition,
            lambda: weights['cosine'] - 0.02,
            lambda: weights['cosine']
        )
        
        avg_pos_weight = tf.cond(
            noise_level_condition,
            lambda: weights['avg_pos'] - 0.01,
            lambda: weights['avg_pos']
        )
        
        std_weight = tf.cond(
            noise_level_condition,
            lambda: weights['std'] - 0.01,
            lambda: weights['std']
        )

        # Modify the faint voice detection logic
        cosine_lower_bound = tf.constant(0.08, dtype=tf.float32)
        cosine_upper_bound = tf.constant(0.25, dtype=tf.float32)
        cosine_middle_bound = tf.constant(0.12, dtype=tf.float32)
        cosine_ratio = tf.constant(0.85, dtype=tf.float32)
        
        faint_voice_condition = tf.logical_and(
            tf.greater(cosine_sim, cosine_lower_bound),
            tf.less(cosine_sim, cosine_upper_bound)
        )
        
        std_weight_adjusted = tf.cond(
            faint_voice_condition,
            lambda: std_weight * 0.7,
            lambda: std_weight
        )
        
        boost_condition = tf.logical_and(
            tf.greater(avg_pos_sim, cosine_ratio * cosine_sim),
            tf.greater(cosine_sim, cosine_middle_bound)
        )
        
        boost = tf.cond(
            tf.logical_and(faint_voice_condition, boost_condition),
            lambda: tf.constant(0.03, dtype=tf.float32),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )

        # Calculate final score
        final_score = (
            cosine_weight * cosine_sim +
            avg_pos_weight * avg_pos_sim +
            gaussian_weight * gaussian_sim -
            weights['negative'] * negative_penalty -
            std_weight_adjusted * std_penalty + 
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
        
        # Ensure score is between 0 and 1
        final_score = tf.clip_by_value(final_score, 0, 1)
        
        return final_score, metrics

    def is_wake_word(self, query_embedding, noise_level=0, threshold=None):
        """Determine if the query embedding represents the wake word"""
        similarity, metrics = self.compute_enhanced_similarity(query_embedding, noise_level)
        
        if threshold is None:
            threshold = tf.constant(0.61, dtype=tf.float32)
        else:
            threshold = tf.constant(threshold, dtype=tf.float32)
        
        return tf.greater(similarity, threshold), similarity, metrics

    def estimate_noise_level(self, audio):
        """Estimate noise level in audio signal"""
        signal_power = tf.reduce_mean(tf.square(audio))
        peak_power = tf.reduce_max(tf.square(audio))
        
        # Avoid division by zero
        peak_power_safe = tf.maximum(peak_power, 1e-10)
        
        # Calculate SNR and noise level
        snr = 10 * tf.math.log(peak_power_safe / signal_power) / tf.math.log(tf.constant(10.0))
        
        # Apply sigmoid function to map SNR to noise level
        noise_level = 1 / (1 + tf.exp(0.1 * (snr - 10)))
        
        # Clip values to range [0, 1]
        return tf.clip_by_value(noise_level, 0, 1)

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

# 1. First, make sure your model has proper input/output signatures
def prepare_model_for_export(model):
    """Prepare the model for TFLite conversion by providing concrete input shapes"""
    
    # Define the concrete input shape for your model
    # Assuming the input is an embedding vector of size 512 (adjust as needed)
    embedding_dimension = 2048  # Change this to match your embedding dimension
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, embedding_dimension], dtype=tf.float32),  # Query embedding shape
        tf.TensorSpec(shape=[], dtype=tf.float32)  # Noise level (scalar)
    ])
    def serving_function(query_embedding, noise_level):
        # Call is_wake_word with fixed threshold
        is_wake, similarity, metrics = model.is_wake_word(query_embedding, noise_level)
        
        # Return what you need in your Android app
        return {
            'is_wake_word': is_wake,
            'similarity_score': similarity
        }
    
    # Save the signatures
    model.serving_function = serving_function
    return model

# 2. Convert the model to TFLite
def convert_to_tflite(model, output_path="enhanced_similarity_matcher.tflite"):
    """Convert the TensorFlow model to TFLite format"""
    
    # Prepare model
    model = prepare_model_for_export(model)
    
    # Get concrete function
    concrete_func = model.serving_function.get_concrete_function()
    
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # Optimization settings (optional)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]  # Use FP16 quantization
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model to file
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model successfully converted and saved to {output_path}")
    return output_path

# Example usage:
# matcher = EnhancedSimilarityMatcher(positive_embeddings, negative_embeddings)
# tflite_path = convert_to_tflite(matcher)



if __name__ == "__main__":
    
    model = ONNXtoTorchModel("./resnet_50_arc/slim_93%_accuracy_72.7390%.onnx")
    
    base_dir = "./"
    
    dir_list = os.listdir(os.path.join(base_dir, "wake_word_data", "recordings"))
    
    positive_files = [
        os.path.join(base_dir, r"tts_samples\positive\Nobita_en-AU-jimm.mp3"),
        os.path.join(base_dir, r"tts_samples\positive\Nobita_en-AU-kylie.mp3"),
        os.path.join(base_dir, r"tts_samples\positive\Nobita_en-IN-aarav.mp3"),
        os.path.join(base_dir, r"tts_samples\positive\Nobita_en-IN-alia.mp3"),
        os.path.join(base_dir, r"tts_samples\positive\Nobita_en-UK-ruby.mp3"),
        os.path.join(base_dir, r"tts_samples\positive\Nobita_en-UK-theo.mp3"),
        os.path.join(base_dir, r"tts_samples\positive\Nobita_en-US-natalie.mp3"),
        os.path.join(base_dir, r"tts_samples\positive\Nobita_en-US-zion.mp3"),
        # os.path.join(base_dir, "tts_samples", "negative", f"Aira1.mp3"),
        # os.path.join(base_dir, "tts_samples", "negative", f"Aira0.mp3"),
    ]

    negative_files = [
        os.path.join(base_dir, "tts_samples", "negative", "Hello0.mp3"),
        os.path.join(base_dir, "tts_samples", "negative", "Hello1.mp3"),
        os.path.join(base_dir, "tts_samples", "negative", "Thunderbolt_en-IN-aarav.mp3"),
        os.path.join(base_dir, "tts_samples", "negative", "Thunderbolt_en-IN-alia.mp3"),
        os.path.join(base_dir, "tts_samples", "negative", "Thunderbolt_en-US-zion.mp3"),
        os.path.join(base_dir, "tts_samples", "negative", "Thunderbolt_en-US-natalie.mp3"),
        os.path.join(base_dir, "tts_samples", "negative", "Xylophone_en-IN-aarav.mp3"),
        os.path.join(base_dir, "tts_samples", "negative", "Xylophone_en-IN-alia.mp3"),
        os.path.join(base_dir, "tts_samples", "negative", "Xylophone_en-US-zion.mp3"),
        os.path.join(base_dir, "tts_samples", "negative", "Xylophone_en-US-natalie.mp3"),
        os.path.join(base_dir, "tts_samples", "negative", "Quasar_en-IN-alia.mp3"),
        os.path.join(base_dir, "tts_samples", "negative", "Quasar_en-IN-aarav.mp3"),
        os.path.join(base_dir, "tts_samples", "negative", "Quasar_en-US-zion.mp3"),
        os.path.join(base_dir, "tts_samples", "negative", "Quasar_en-US-natalie.mp3"),
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
    
    print(matcher)    
    
    file_path = r"C:\Users\Rohit Francis\Documents\GitHub\EfficientWordNet_Upgrade\tts_samples\positive\Nobita_en-AU-jimm.mp3"
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
    # noise_level = matcher.estimate_noise_level(tf.constant(audio))
    check, similarity, metrics = matcher.call(embeddings_audio, 0.0)

    print(check)
    print(similarity)
    print(metrics)
    
    tflite_path = convert_to_tflite(matcher)
    print(f"Model successfully converted and saved to {tflite_path}")
    
    # Load and test TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], 
                           tf.reshape(embeddings_audio, [1, -1]).numpy())
    interpreter.set_tensor(input_details[1]['index'], 
                           np.array(0.0, dtype=np.float32))
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    tflite_is_wake = interpreter.get_tensor(output_details[0]['index'])
    tflite_score = interpreter.get_tensor(output_details[1]['index'])
    
    print(f"TFLite model output - Is wake word: {tflite_is_wake}, Score: {tflite_score}")
    
# import pyttsx3
# import time
# engine = pyttsx3.init()  # Initialize the TTS engine
# word = "Nobheetha"
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[0].id)
# engine.say(word)  # Queue text for speaking
# engine.save_to_file(word, rf"C:\Users\Rohit Francis\Documents\GitHub\EfficientWordNet_Upgrade\tts_samples\negative\{word}0.mp3")
# engine.runAndWait()  # Process the speech

# engine.setProperty('voice', voices[1].id)
# engine.say(word)  # Queue text for speaking
# engine.runAndWait()  # Process the speech
# engine.save_to_file(word, rf"C:\Users\Rohit Francis\Documents\GitHub\EfficientWordNet_Upgrade\tts_samples\negative\{word}1.mp3")
# engine.runAndWait()  # Process the speech
# print("All saved")
# import os
# import nltk
# import subprocess
# import re
# import glob
# import random
# from nltk.corpus import cmudict
# from jellyfish import jaro_winkler_similarity, levenshtein_distance
# import numpy as np

# # Download CMU Pronouncing Dictionary if not already present
# nltk.download('cmudict', quiet=True)
# cmu_dict = cmudict.dict()

# def get_cmu_phonemes(word):
#     """Fetch phonetic transcription from CMU Pronouncing Dictionary."""
#     word = word.lower()
#     if word in cmu_dict:
#         return cmu_dict[word][0]  # CMU dictionary may have multiple pronunciations
#     return None

# def get_espeak_phonemes(word):
#     """Fetch phonetic transcription using espeak-ng."""
#     try:
#         result = subprocess.run(["espeak-ng", "--ipa", word], capture_output=True, text=True)
#         return result.stdout.strip()
#     except Exception as e:
#         print(f"Error with espeak-ng: {e}")
#         return None

# def calculate_phonetic_similarity(word1, word2):
#     """Calculate phonetic similarity between two words using multiple metrics."""
#     # Convert to lowercase
#     word1, word2 = word1.lower(), word2.lower()
    
#     # Direct string similarity
#     string_similarity = jaro_winkler_similarity(word1, word2)
    
#     # Levenshtein distance (normalized)
#     max_len = max(len(word1), len(word2))
#     levenshtein_sim = 1 - (levenshtein_distance(word1, word2) / max_len if max_len > 0 else 0)
    
#     # Phonetic similarity using CMU dict
#     phonemes1 = get_cmu_phonemes(word1)
#     phonemes2 = get_cmu_phonemes(word2)
    
#     phonetic_sim = 0
#     if phonemes1 and phonemes2:
#         # Calculate similarity based on phonemes
#         # Count matching phonemes in sequence
#         matches = 0
#         total = max(len(phonemes1), len(phonemes2))
        
#         for i in range(min(len(phonemes1), len(phonemes2))):
#             # Count as partial match if the consonant/vowel part matches
#             if phonemes1[i][0] == phonemes2[i][0]:
#                 matches += 0.5
#                 # If the entire phoneme matches
#                 if phonemes1[i] == phonemes2[i]:
#                     matches += 0.5
        
#         phonetic_sim = matches / total if total > 0 else 0
    
#     # Combine metrics (weighted average)
#     combined_similarity = (0.4 * string_similarity + 
#                           0.3 * levenshtein_sim + 
#                           0.3 * phonetic_sim)
    
#     return combined_similarity

# def find_wake_word_files(base_dir, pattern=None):
#     """Find all wake word audio files in the base directory."""
#     wake_word_files = []
    
#     # Path to recordings directory
#     recordings_dir = os.path.join(base_dir, "wake_word_data", "recordings")
    
#     # Search types (subdirectories)
#     types = ["normal", "quick", "shouted", "whispered"]
    
#     for type_dir in types:
#         search_path = os.path.join(recordings_dir, type_dir, "*.wav")
#         files = glob.glob(search_path)
#         wake_word_files.extend(files)
    
#     # Extract unique wake words from filenames
#     wake_words = set()
#     for file_path in wake_word_files:
#         filename = os.path.basename(file_path)
#         # Extract wake word from filename (format: WakeWord_type_num.wav)
#         wake_word = filename.split('_')[0]
#         wake_words.add(wake_word)
    
#     return list(wake_words), wake_word_files

# def get_similarity_groups(target_word, wake_words, num_similar=2, num_dissimilar=2, num_neutral=2):
#     """Group wake words based on their similarity to the target word."""
#     similarities = []
    
#     for word in wake_words:
#         if word.lower() == target_word.lower():
#             continue  # Skip the target word itself
        
#         similarity = calculate_phonetic_similarity(target_word, word)
#         similarities.append((word, similarity))
    
#     # Sort by similarity score
#     similarities.sort(key=lambda x: x[1], reverse=True)
    
#     # Group by similarity level
#     high_similarity = [word for word, score in similarities[:num_similar]]
#     low_similarity = [word for word, score in similarities[-num_dissimilar:]]
    
#     # Pick some from the middle for neutral similarity
#     if len(similarities) > num_similar + num_dissimilar:
#         middle_idx = len(similarities) // 2
#         start_idx = max(num_similar, middle_idx - num_neutral // 2)
#         neutral_similarity = [word for word, score in 
#                              similarities[start_idx:start_idx + num_neutral]]
#     else:
#         neutral_similarity = []
    
#     return {
#         "high_similarity": high_similarity,
#         "neutral_similarity": neutral_similarity,
#         "low_similarity": low_similarity
#     }

# def get_negative_files(base_dir, target_word, similarity_groups, files_per_group=2):
#     """Create a list of negative files based on similarity groups."""
#     negative_files = []
#     all_wake_words, all_files = find_wake_word_files(base_dir)
    
#     # Group files by wake word
#     word_to_files = {}
#     for file_path in all_files:
#         filename = os.path.basename(file_path)
#         wake_word = filename.split('_')[0]
        
#         if wake_word not in word_to_files:
#             word_to_files[wake_word] = []
        
#         word_to_files[wake_word].append(file_path)
    
#     # Add files from each similarity group
#     for group_name, words in similarity_groups.items():
#         for word in words:
#             if word in word_to_files:
#                 # Get a random sample of files for this word
#                 word_files = word_to_files[word]
#                 sample_size = min(files_per_group, len(word_files))
#                 selected_files = random.sample(word_files, sample_size)
#                 negative_files.extend(selected_files)
    
#     return negative_files

# def get_positive_files(base_dir, target_word, variations=None):
#     """Get positive files for the target wake word."""
#     positive_files = []
#     all_wake_words, all_files = find_wake_word_files(base_dir)
    
#     # Filter files for the target word
#     for file_path in all_files:
#         filename = os.path.basename(file_path)
#         if filename.startswith(f"{target_word}_"):
#             # If variations are specified, check if this file matches
#             if variations:
#                 for variation in variations:
#                     if f"_{variation}_" in filename:
#                         positive_files.append(file_path)
#                         break
#             else:
#                 positive_files.append(file_path)
    
#     return positive_files

# def search_wake_words(base_dir, target_word, high_similar=2, neutral=2, low_similar=2, files_per_group=2):
#     """Main function to search for wake words based on phonetic similarity."""
#     # Find all available wake words
#     all_wake_words, all_files = find_wake_word_files(base_dir)
    
#     print(f"\n🔍 Searching for wake words similar to '{target_word}'")
#     print(f"Found {len(all_wake_words)} unique wake words in the dataset")
    
#     # Group wake words by similarity
#     similarity_groups = get_similarity_groups(
#         target_word, all_wake_words, 
#         num_similar=high_similar,
#         num_neutral=neutral,
#         num_dissimilar=low_similar
#     )
    
#     # Print similarity groups
#     print("\n📊 Similarity Analysis:")
#     for group, words in similarity_groups.items():
#         print(f"\n{group.replace('_', ' ').title()}:")
#         for word in words:
#             similarity = calculate_phonetic_similarity(target_word, word)
#             print(f"  • {word} (similarity score: {similarity:.2f})")
    
#     # Get the positive and negative files
#     positive_files = get_positive_files(base_dir, target_word)
#     negative_files = get_negative_files(base_dir, target_word, similarity_groups, files_per_group)
    
#     result = {
#         "target_word": target_word,
#         "similarity_groups": similarity_groups,
#         "positive_files": positive_files,
#         "negative_files": negative_files
#     }
    
#     return result

# # Example usage
# if __name__ == "__main__":
#     base_dir = "."  # Replace with your actual base directory
#     target_word = "Ava"
    
#     result = search_wake_words(base_dir, target_word)
    
#     print("\n✅ Positive Files:")
#     for file in result["positive_files"][:5]:  # Show first 5 for brevity
#         print(f"  • {os.path.basename(file)}")
    
#     print("\n❌ Negative Files (selected based on similarity):")
#     for file in result["negative_files"][:10]:  # Show first 10 for brevity
#         print(f"  • {os.path.basename(file)}")

# import nltk
# import subprocess
# from nltk.corpus import cmudict

# # Download CMU Pronouncing Dictionary if not already present
# nltk.download('cmudict')
# cmu_dict = cmudict.dict()

# def get_cmu_phonemes(word):
#     """Fetch phonetic transcription from CMU Pronouncing Dictionary."""
#     word = word.lower()
#     if word in cmu_dict:
#         return cmu_dict[word][0]  # CMU dictionary may have multiple pronunciations
#     return None

# def get_espeak_phonemes(word):
#     """Fetch phonetic transcription using espeak-ng."""
#     try:
#         result = subprocess.run(["espeak-ng", "--ipa", word], capture_output=True, text=True)
#         return result.stdout.strip()
#     except Exception as e:
#         print(f"Error with espeak-ng: {e}")
#         return None

# # Example Usage
# words = ["Alexa", "Lexi", "Banana", "Google", "Jarvis", "Salo", "Ava"]

# for word in words:
#     cmu_phonemes = get_cmu_phonemes(word)
#     # espeak_phonemes = get_espeak_phonemes(word)
    
#     print(f"\nWord: {word}")
#     print(f"CMU Phonemes: {cmu_phonemes if cmu_phonemes else 'Not Found'}")
#     # print(f"Espeak-ng IPA: {espeak_phonemes if espeak_phonemes else 'Not Found'}")


# """
# Can be run directly in cli 
# `python -m generate_reference`
# """
# from rich.progress import track
# import typer
# import librosa
# import os
# import glob
# import numpy as np
# import json
# from package_installation_scripts import check_install_librosa
# from audio_processing import (
#     ModelType,
#     MODEL_TYPE_MAPPER
# )

# check_install_librosa()


# def generate_reference_file(
#     input_dir: str = typer.Option(...),
#     output_dir: str = typer.Option(...),
#     wakeword: str = typer.Option(...),
#     model_type: ModelType = typer.Option(..., case_sensitive=False),
#     debug: bool = typer.Option(False)
# ):
#     """
#     Generates reference files for few shot learning comparison

#     Inp Parameters:

#         input_dir : directory which holds only sample audio files
#         of wakeword

#         output_dir: directory where generated reference file will
#         be stored

#         wakeword: name of the wakeword

#         model_type: type of the model to be used

#         debug: self explanatory
#     Out Parameters:

#         None

#     """
#     # print(model_type)
#     model = MODEL_TYPE_MAPPER[model_type.value]()

#     assert (os.path.isdir(input_dir))
#     assert (os.path.isdir(output_dir))
#     embeddings = []

#     audio_files = [
#         *glob.glob(input_dir+"/*.mp3"),
#         *glob.glob(input_dir+"/*.wav")
#     ]

#     for audio_file in track(audio_files, description="Generating Embeddings.. "):
#         x, _ = librosa.load(audio_file, sr=16000)
#         embeddings.append(
#             model.audioToVector(
#                 model.fixPaddingIssues(x)
#             )
#         )

#     embeddings = np.squeeze(np.array(embeddings))

#     if (debug):
#         distanceMatrix = []

#         for embedding in embeddings:
#             distanceMatrix.append(
#                 np.sqrt(np.sum((embedding-embeddings)**2, axis=1))
#             )

#         temp = np.squeeze(distanceMatrix).astype(np.float16)
#         temp2 = temp.flatten()
#         print(np.std(temp2), np.mean(temp2))
#         print(temp)
        
#     negative_files = [
#             os.path.join(base_dir, "tts_samples", "negative", "partial_voice0.wav"),
#             os.path.join(base_dir, "tts_samples", "negative", "partial_voice1.wav"),
#             os.path.join(base_dir, "tts_samples", "negative", "last_part_voice0.wav"),
#             os.path.join(base_dir, "tts_samples", "negative", "last_part_voice1.wav")
#     ]
    
#     print(f"{Fore.RED}Processing negative examples...{Style.RESET_ALL}")
#     negative_embeddings = []
#     for file in negative_files:
        
#         audio, sr = librosa.load(file, sr=16000)
#         # Ensure audio is exactly 24000 samples long
#         expected_length = 24000
#         if len(audio) < expected_length:
#             pad_length = expected_length - len(audio)
#             audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros
        
#         emb = model(audio)
#         negative_embeddings.append(emb.detach().tolist()[0])


#     open(os.path.join(output_dir, f"{wakeword}_ref.json"), 'w').write(
#         json.dumps(
#             {
#                 "positive_embeddings": embeddings.astype(float).tolist(),
#                 "negative_embeddings": negative_embeddings,
#                 "model_type": model_type.value
#             }
#         )
#     )


# if __name__ == "__main__":
#     typer.run(generate_reference_file)



# Test code
# import json
# import librosa
# from colorama import Fore, Style
# from Detection import ONNXtoTorchModel
# import os
# import numpy as np

# word = "SnehaAIGen"
# negative_word = "none" 

# base_dir = "./"

# model_path = os.path.join(base_dir, "resnet_50_arc", "slim_93%_accuracy_72.7390%.onnx")
# model = ONNXtoTorchModel(model_path)


# positive_files = [
#     os.path.join(base_dir, r"tts_samples\positive\Sneha_en-AU-jimm.mp3"),
#     os.path.join(base_dir, r"tts_samples\positive\Sneha_en-AU-kylie.mp3"),
#     os.path.join(base_dir, r"tts_samples\positive\Sneha_en-IN-aarav.mp3"),
#     os.path.join(base_dir, r"tts_samples\positive\Sneha_en-IN-alia.mp3"),
#     os.path.join(base_dir, r"tts_samples\positive\Sneha_en-UK-ruby.mp3"),
#     os.path.join(base_dir, r"tts_samples\positive\Sneha_en-UK-theo.mp3"),
#     os.path.join(base_dir, r"tts_samples\positive\Sneha_en-US-natalie.mp3"),
#     os.path.join(base_dir, r"tts_samples\positive\Sneha_en-US-zion.mp3"),
#     os.path.join(base_dir, "wake_word_data", "recordings", "normal", f"Sneha_normal_1.wav"),
#     os.path.join(base_dir, "wake_word_data", "recordings", "normal", f"Sneha_normal_2.wav"),
#     # os.path.join(base_dir, "tts_samples", "negative", f"Aira1.mp3"),
#     # os.path.join(base_dir, "tts_samples", "negative", f"Aira0.mp3"),
# ]


# positive_files = [
#     os.path.join(base_dir, "wake_word_data", "recordings", "normal", f"{word}_normal_1.wav"),
#     os.path.join(base_dir, "wake_word_data", "recordings", "quick", f"{word}_quick_1.wav"),
#     os.path.join(base_dir, "wake_word_data", "recordings", "shouted", f"{word}_shouted_1.wav"),
#     os.path.join(base_dir, "wake_word_data", "recordings", "whispered", f"{word}_whispered_1.wav"),
#     os.path.join(base_dir, "tts_samples", "negative", f"{word}1.mp3"),
#     os.path.join(base_dir, "tts_samples", "negative", f"{word}0.mp3"),
#     # os.path.join(base_dir, "tts_samples", "negative", f"Aira1.mp3"),
#     # os.path.join(base_dir, "tts_samples", "negative", f"Aira0.mp3"),
# ]

# negative_files = [
#     os.path.join(base_dir, "tts_samples", "negative", "partial_voice0.wav"),
#     os.path.join(base_dir, "tts_samples", "negative", "partial_voice1.wav"),
#     os.path.join(base_dir, "tts_samples", "negative", "last_part_voice0.wav"),
#     os.path.join(base_dir, "tts_samples", "negative", "last_part_voice1.wav"),
# #      os.path.join(base_dir, "tts_samples", "negative", f"{negative_word}1.mp3"),
# #      os.path.join(base_dir, "tts_samples", "negative", f"{negative_word}0.mp3")
# ]

# negative_files = [
#     os.path.join(base_dir, "tts_samples", "negative", "Hello0.mp3"),
#     os.path.join(base_dir, "tts_samples", "negative", "Hello1.mp3"),
#     os.path.join(base_dir, "tts_samples", "negative", "Xylophone0.mp3"),
#     os.path.join(base_dir, "tts_samples", "negative", "Xylophone1.mp3"),
#     os.path.join(base_dir, "tts_samples", "negative", f"Thunderbolt1.mp3"),
#     os.path.join(base_dir, "tts_samples", "negative", f"Thunderbolt0.mp3"),
#     os.path.join(base_dir, "tts_samples", "negative", f"Quasar1.mp3"),
#     os.path.join(base_dir, "tts_samples", "negative", f"Quasar0.mp3"),
#     os.path.join(base_dir, "tts_samples", "negative", f"{negative_word}1.mp3"),
#     os.path.join(base_dir, "tts_samples", "negative", f"{negative_word}0.mp3")
# ]

# negative_files = [
#         os.path.join(base_dir, "wake_word_data", "recordings", "normal", "Hello_normal_1.wav"),
#         os.path.join(base_dir, "wake_word_data", "recordings", "quick", "Hello_quick_1.wav"),
#         os.path.join(base_dir, "wake_word_data", "recordings", "normal", "Ham_normal_2.wav"),
#         os.path.join(base_dir, "wake_word_data", "recordings", "shouted", "Ham_shouted_3.wav"),
#         os.path.join(base_dir, "wake_word_data", "recordings", "whispered", "Ham_whispered_3.wav"),
#         os.path.join(base_dir, "wake_word_data", "recordings", "normal", "Jeeva_normal_1.wav"),
#     ]

# negative_files = [
#         # os.path.join(base_dir, "tts_samples", "negative", "Hello0.mp3"),
#         # os.path.join(base_dir, "tts_samples", "negative", "Hello1.mp3"),
#         os.path.join(base_dir, "tts_samples", "negative", "Thunderbolt_en-IN-aarav.mp3"),
#         os.path.join(base_dir, "tts_samples", "negative", "Thunderbolt_en-IN-alia.mp3"),
#         os.path.join(base_dir, "tts_samples", "negative", "Thunderbolt_en-US-zion.mp3"),
#         os.path.join(base_dir, "tts_samples", "negative", "Thunderbolt_en-US-natalie.mp3"),
#         os.path.join(base_dir, "tts_samples", "negative", "Xylophone_en-IN-aarav.mp3"),
#         os.path.join(base_dir, "tts_samples", "negative", "Xylophone_en-IN-alia.mp3"),
#         os.path.join(base_dir, "tts_samples", "negative", "Xylophone_en-US-zion.mp3"),
#         os.path.join(base_dir, "tts_samples", "negative", "Xylophone_en-US-natalie.mp3"),
#         os.path.join(base_dir, "tts_samples", "negative", "Quasar_en-IN-alia.mp3"),
#         os.path.join(base_dir, "tts_samples", "negative", "Quasar_en-IN-aarav.mp3"),
#         os.path.join(base_dir, "tts_samples", "negative", "Quasar_en-US-zion.mp3"),
#         os.path.join(base_dir, "tts_samples", "negative", "Quasar_en-US-natalie.mp3"),
#     ]
# Old negative reference
# negative_files = [
#         os.path.join(base_dir, "wake_word_data", "recordings", "normal", "Hello_normal_1.wav"),
#         os.path.join(base_dir, "wake_word_data", "recordings", "quick", "Hello_quick_1.wav"),
#         os.path.join(base_dir, "wake_word_data", "recordings", "normal", "Eliza_normal_2.wav"),
#         os.path.join(base_dir, "wake_word_data", "recordings", "shouted", "Eliza_shouted_3.wav"),
#         os.path.join(base_dir, "wake_word_data", "recordings", "whispered", "Eliza_whispered_3.wav"),
#         os.path.join(base_dir, "wake_word_data", "recordings", "normal", "Jeeva_normal_1.wav"),
#     ]
    
# Process positive examples
# print(f"{Fore.GREEN}Processing positive examples...{Style.RESET_ALL}")
# positive_embeddings = []
# for file in positive_files:
    
#     audio, sr = librosa.load(file, sr=16000)
#     # Ensure audio is exactly 24000 samples long
#     expected_length = 24000
#     if len(audio) < expected_length:
#         pad_length = expected_length - len(audio)
#         audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros
    
#     emb = model(audio)
#     positive_embeddings.append(emb.detach().tolist()[0])

# # Process negative examples
# print(f"{Fore.RED}Processing negative examples...{Style.RESET_ALL}")
# negative_embeddings = []
# for file in negative_files:
    
#     audio, sr = librosa.load(file, sr=16000)
#     # Ensure audio is exactly 24000 samples long
#     expected_length = 24000
#     if len(audio) < expected_length:
#         pad_length = expected_length - len(audio)
#         audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros
    
#     emb = model(audio)
#     negative_embeddings.append(emb.detach().tolist()[0])


# # print(positive_embeddings[0])

# data = {"positive_embeddings":positive_embeddings, 
#         "negative_embeddings": negative_embeddings}

# with open(f"{word}_{negative_word}_ref.json", "w") as f:
#     s = json.dumps(data)
#     f.write(s)

# print(f"saved {word}_{negative_word}_ref.json")
# {
#     "positive_embeddings": [
#         [0.02489, 0.0471, ...],
#         [0.009754, 0.004944, ...]
#     ],
#     "negative_embeddings": [
#         [0.02489, 0.0471, ...],
#         [0.009754, 0.004944, ...]
#     ]
# }


# import os
# import numpy as np
# from Detection import ONNXtoTorchModel
# import numpy as np
# from python_speech_features import logfbank

# class ModelRawBackend :
#     def __init__(self, use_quantized_model=False):
#         self.use_quantized_model = use_quantized_model
#         self.window_length = None
#         self.window_frames = None
#         pass 

#     def _randomCrop(self, x:np.array,length=16000)->np.array :
#         assert(x.shape[0]>self.window_frames)
#         frontBits = random.randint(0,x.shape[0]-length) 
#         return x[frontBits:frontBits+length]
    
#     def _addPadding(self, x:np.array,length=16000)->np.array :
#         assert(x.shape[0]<length)
#         bitCountToBeAdded = length - x.shape[0]
#         frontBits = random.randint(0,bitCountToBeAdded)
#         #print(frontBits, bitCountToBeAdded-frontBits)
#         new_x = np.append(np.zeros(frontBits),x)
#         new_x = np.append(new_x,np.zeros(bitCountToBeAdded-frontBits))
#         return new_x
    
#     def _removeExistingPadding(self, x:np.array)->np.array:
#         lastZeroBitBeforeAudio = 0 
#         firstZeroBitAfterAudio = len(x)
#         for i in range(len(x)):
#           if x[i]==0:
#             lastZeroBitBeforeAudio = i
#           else:
#             break
#         for i in range(len(x)-1,1,-1):
#           if x[i]==0:
#             firstZeroBitAfterAudio = i
#           else:
#             break
#         return x[lastZeroBitBeforeAudio:firstZeroBitAfterAudio]
    
#     def fixPaddingIssues(self, x:np.array)-> np.array:
#         x = self._removeExistingPadding(x)
#         #print("Preprocessing Shape",x.shape[0])
#         if(x.shape[0]>self.window_frames):
#           return self._randomCrop(x,length=self.window_frames)
#         elif(x.shape[0]<self.window_frames):
#           return self._addPadding(x,length=self.window_frames)
#         else:
#           return x

#     def scoreVector(self, inp_vector:np.array, embeddings:np.array) -> np.array:
#         raise NotImplementedError("Vector scoring attempted on raw model backend")
    
#     def audioToVector(self, inpAudio:np.array) -> np.array :
#         raise NotImplementedError("Vector Convertion on raw model backend invoked")


# class Resnet50_Arc_loss(ModelRawBackend):
#     def __init__(self):
#         super().__init__()
        
#         self.window_length = 1.5
#         self.window_frames = int(self.window_length * 16000)

#         self.onnx_sess = ONNXtoTorchModel("./resnet_50_arc/slim_93%_accuracy_72.7390%.onnx")
        
#         self.input_name:str = self.onnx_sess.get_inputs()[0].name
#         self.output_name:str = self.onnx_sess.get_outputs()[0].name

#         self.audioToVector(np.float32(np.zeros(self.window_frames,))) #warmup inference

#     def compute_logfbank_features(self, inpAudio:np.array)->np.array:
#         """
#         This assumes a mono channel input
#         """
#         return logfbank(
#            inpAudio,
#            samplerate=16000,
#            winlen=0.025,
#            winstep=0.01,
#            nfilt=64,
#            nfft=512,
#            preemph=0.0
#         )
    
#     def scoreVector(self, inp_vector: np.array, embeddings: np.array) -> np.array:
#         #print(inp_vector.shape, embeddings.shape)
#         #inp_norm = np.sqrt(np.sum(inp_vector**2, axis=1))
#         #embeddings_norm = np.sqrt(np.sum(embeddings**2, axis=1))
#         #print(inp_norm, embeddings_norm)

#         #inp_vector = inp_vector/  inp_norm
#         #embeddings = embeddings/ np.expand_dims(embeddings_norm, axis = -1)

#         #inp_vector = inp_vector / np.linalg.norm(inp_vector)
#         #embeddings = embeddings / np.expand_dims(np.linalg.norm(embeddings, axis=0), axis = -1)

#         cosine_similarity = np.matmul(embeddings, inp_vector.T)
#         #print(cosine_similarity)
#         confidence_scores = (cosine_similarity+1)/2
#         #print(confidence_scores.max())
#         #print(cosine_similarity.shape, cosine_similarity.max())

#         return confidence_scores.max()

#     def audioToVector(self, inpAudio: np.array) -> np.array:
        
#         assert inpAudio.shape == (self.window_frames, ) #1.5 sec long window
#         features = self.compute_logfbank_features(inpAudio)
#         #features_norm = self.min_max_normalize_features(features)

#         output = self.onnx_sess.run(
#            [self.output_name],
#            {
#                 self.input_name : np.float32(
#                     np.expand_dims(
#                         features,
#                         #features_norm,
#                         axis = (0,1) # adding channel and batch dimension
#                     )
#                 )
#            }
#         )[0]
        
#         return output
    

# import os
# import numpy as np
# import librosa
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy.stats import norm
# from colorama import Fore, Style

# def tabulate(headers, results):
#     """
#     Prints a formatted table without using external libraries.
    
#     :param headers: List of column headers
#     :param results: List of row data
#     """
#     col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *results)]
    
#     def format_row(row):
#         return " | ".join(f"{str(item):<{col_widths[i]}}" for i, item in enumerate(row))
    
#     print(f"\n{Fore.CYAN}=== Test Results ==={Style.RESET_ALL}")
#     print("-" * (sum(col_widths) + 3 * (len(headers) - 1)))
#     print(format_row(headers))
#     print("-" * (sum(col_widths) + 3 * (len(headers) - 1)))
#     for row in results:
#         print(format_row(row))
#     print("-" * (sum(col_widths) + 3 * (len(headers) - 1)))

# class EnhancedSimilarityMatcher:
#     def __init__(self, positive_embeddings, negative_embeddings=None, noise_levels=None):
#         """
#         Initialize with positive and optional negative example embeddings
#         """
#         # Convert embeddings to numpy and ensure correct shape
#         self.positive_embeddings = np.array([emb.squeeze().cpu().numpy() for emb in positive_embeddings])
#         self.negative_embeddings = np.array([emb.squeeze().cpu().numpy() for emb in negative_embeddings]) if negative_embeddings else None
#         self.noise_levels = np.array(noise_levels) if noise_levels else None
        
#         # Calculate statistics from positive examples
#         self.positive_centroid = np.mean(self.positive_embeddings, axis=0)
#         self.positive_std = np.std(self.positive_embeddings, axis=0)
        
#         self.old_model = Resnet50_Arc_loss()
        
#         # Calculate decision boundaries if negative samples exist
#         if self.negative_embeddings is not None:
#             self.negative_centroid = np.mean(self.negative_embeddings, axis=0)
#             self._calculate_decision_boundary()
    
#     def _calculate_decision_boundary(self):
#         """Calculate optimal decision boundary using positive and negative samples"""
#         pos_sims = self._batch_cosine_similarity(self.positive_embeddings, self.positive_centroid)
#         neg_sims = self._batch_cosine_similarity(self.negative_embeddings, self.positive_centroid)
        
#         # Find optimal threshold that maximizes separation
#         all_sims = np.concatenate([pos_sims, neg_sims])
#         all_sims.sort()
        
#         best_threshold = 0
#         best_separation = -float('inf')
        
#         for threshold in all_sims:
#             pos_correct = np.mean(pos_sims >= threshold)
#             neg_correct = np.mean(neg_sims < threshold)
#             separation = pos_correct + neg_correct - 1
#             if separation > best_separation:
#                 best_separation = separation
#                 best_threshold = threshold
        
#         self.decision_threshold = best_threshold
    
#     def _batch_cosine_similarity(self, embeddings, reference):
#         """Calculate cosine similarity between embeddings and reference"""
#         # Ensure shapes are correct
#         if len(embeddings.shape) == 3:
#             embeddings = embeddings.squeeze(1)
#         if len(reference.shape) == 1:
#             reference = reference.reshape(1, -1)
            
#         return cosine_similarity(embeddings, reference).flatten()
    
#     def _adaptive_gaussian_kernel(self, distance, noise_level=0):
#         """Compute Gaussian kernel with adaptive sigma based on noise level"""
#         base_sigma = 0.4  # Increased to be more lenient with variations
#         max_sigma = 0.6  # Increased accordingly
        
#         adaptive_sigma = base_sigma + (max_sigma - base_sigma) * noise_level
#         return norm.pdf(distance, loc=0, scale=adaptive_sigma)
    
#     def compute_enhanced_similarity(self, query_embedding, noise_level=0):
#         """Compute enhanced similarity score using multiple metrics"""
#         # Ensure query_embedding is the right shape
#         query_embedding = query_embedding.squeeze()
        
#         # 1. Cosine similarity with positive centroid
#         cosine_sim = cosine_similarity(
#             query_embedding.reshape(1, -1), 
#             self.positive_centroid.reshape(1, -1)
#         )[0][0]
        
#         # 2. Average similarity to positive examples
#         pos_similarities = self._batch_cosine_similarity(self.positive_embeddings, query_embedding)
#         avg_pos_sim = np.mean(pos_similarities)
        
#         # 3. Distance from negative samples (if available)
#         negative_penalty = 0
#         if self.negative_embeddings is not None:
#             neg_sims = self._batch_cosine_similarity(self.negative_embeddings, query_embedding)
#             negative_penalty = np.mean(neg_sims)
        
#         # 4. Gaussian kernel similarity with adaptive sigma
#         embedding_distance = np.linalg.norm(query_embedding - self.positive_centroid)
#         gaussian_sim = self._adaptive_gaussian_kernel(embedding_distance, noise_level)
        
#         # 5. Standard deviation check (penalize outliers)
#         std_penalty = np.mean(
#             np.abs(query_embedding - self.positive_centroid) > 2 * self.positive_std
#         )
        
#         # ADJUSTED WEIGHTS based on test results analysis
#         weights = {
#             'cosine': 0.45,      # Keep this as is - it's a strong indicator
#             'avg_pos': 0.35,     # Keep this as is - it's helpful for faint voices
#             'gaussian': 0.15,    # Keep this as is
#             'negative': 0.20,    # Increase from 0.10 to 0.20 - put more weight on negative examples
#             'std': 0.05          # Increase from 0.03 to 0.05 - be more strict about standard deviation
#         }

#         # Adjust the noise level handling
#         if noise_level > 0.3:
#             weights['gaussian'] += 0.05
#             weights['cosine'] -= 0.02
#             weights['avg_pos'] -= 0.01
#             weights['std'] -= 0.01  # Less reduction in std penalty (from -0.02)

#         # Modify the faint voice detection logic
#         if cosine_sim < 0.25 and cosine_sim > 0.08:  # Increase lower bound from 0.05 to 0.08
#             # Potential faint voice - still reduce penalties but less aggressively
#             weights['std'] *= 0.7  # Reduce less (from 0.5 to 0.7)
            
#             # Make the boost more conditional
#             if avg_pos_sim > 0.85 * cosine_sim and cosine_sim > 0.12:
#                 boost = 0.03  # Smaller boost (from 0.05 to 0.03)
#             else:
#                 boost = 0
#         else:
#             boost = 0
        
    
#         final_score = (
#             weights['cosine'] * cosine_sim +
#             weights['avg_pos'] * avg_pos_sim +
#             weights['gaussian'] * gaussian_sim -
#             weights['negative'] * negative_penalty -
#             weights['std'] * std_penalty + 
#             boost
#         )        
#         # Normalize score to [0, 1] range
#         final_score = (final_score + 1) / 2
        
#         # Calculate individual metric scores for detailed analysis
#         metrics = {
#             'cosine_sim': cosine_sim,
#             'avg_pos_sim': avg_pos_sim,
#             'gaussian_sim': gaussian_sim,
#             'negative_penalty': negative_penalty,
#             'std_penalty': std_penalty
#         }
        
#         return np.clip(final_score, 0, 1), metrics  # Ensure score is between 0 and 1
    
#     def is_wake_word(self, query_embedding, noise_level=0, threshold=None):
#         """Determine if the query embedding represents the wake word"""
#         similarity, metrics = self.compute_enhanced_similarity(query_embedding.detach().numpy(), noise_level)
        
#         if threshold is None:
#             # threshold = getattr(self, 'decision_threshold', 0.55)  # Default to 0.55 if no decision_threshold
#             threshold = 0.58
            
#         return similarity > threshold, similarity, metrics

#     def process_audio(self, file_path, model, expected_length=24000):
#         """Process audio file and return embeddings"""
#         print(f"Processing: {os.path.basename(file_path)}")
#         audio, sr = librosa.load(file_path, sr=16000)
#         if len(audio) < expected_length:
#             audio = np.pad(audio, (0, expected_length - len(audio)), mode='constant')
#         audio = audio[:expected_length]
#         return model(audio), audio, sr

#     def estimate_noise_level(self, audio):
#         """Estimate noise level in audio signal"""
#         signal_power = np.mean(audio ** 2)
#         peak_power = np.max(audio ** 2)
        
#         if peak_power > 0:
#             snr = 10 * np.log10(peak_power / signal_power)
#             noise_level = 1 / (1 + np.exp(0.1 * (snr - 10)))
#         else:
#             noise_level = 1.0
#         return np.clip(noise_level, 0, 1)

#     def run_comprehensive_test(self, model, test_files, threshold=None):
#         """
#         Run comprehensive testing with detailed metrics using EnhancedSimilarityMatcher
        
#         :param model: Function to convert audio to embeddings
#         :param test_files: List of test audio files to evaluate
#         :param threshold: Detection threshold (default: None, uses self.decision_threshold or 0.55)
#         :return: List of test results
#         """
#         print(f"\n{Fore.CYAN}=== Enhanced Wake Word Detection System Test ===\n{Style.RESET_ALL}")
        
#         # Test results
#         results = []
#         print(f"\n{Fore.YELLOW}Running tests with EnhancedSimilarityMatcher...{Style.RESET_ALL}")
        
#         for test_file in test_files:
#             # Process audio file
#             emb, audio, sr = self.process_audio(test_file, model)
            
#             # Estimate noise level
#             noise_level = self.estimate_noise_level(audio)
            
#             # Determine if wake word detected using the enhanced similarity logic
#             is_wake, confidence, metrics = self.is_wake_word(emb, noise_level, threshold)
            
#             # Add result with detailed metrics
#             results.append([
#                 os.path.basename(test_file),
#                 f"{confidence:.4f}",
#                 "✓" if is_wake else "✗",
#                 f"{metrics['cosine_sim']:.4f}",
#                 f"{metrics['avg_pos_sim']:.4f}",
#                 f"{metrics['gaussian_sim']:.4f}",
#                 f"{metrics['negative_penalty']:.4f}",
#                 f"{metrics['std_penalty']:.4f}",
#                 f"{noise_level:.4f}"
#             ])
        
#         # Print results table
#         headers = ["File", "Confidence", "Detection", "Cosine", "Avg Pos", "Gaussian", "Neg Penalty", "Std Penalty", "Noise Level"]
#         tabulate(headers, results)
        
#         # Print statistics
#         detections = sum(1 for r in results if r[2] == "✓")
#         print(f"\n{Fore.CYAN}=== Statistics ==={Style.RESET_ALL}")
#         print(f"Total tests: {len(results)}")
#         print(f"Detections: {detections}")
#         print(f"Detection rate: {detections/len(results)*100:.2f}%")
        
#         if hasattr(self, 'decision_threshold'):
#             print(f"Using calculated decision threshold: {self.decision_threshold:.4f}")
#         else:
#             print(f"Using default threshold: {threshold or 0.55:.4f}")
            
#         # Calculate false positive and negative rates if we can infer expected results from filenames
#         positive_tests = sum(1 for file in test_files if "negative" not in os.path.basename(file).lower())
#         negative_tests = len(test_files) - positive_tests
        
#         true_positives = sum(1 for i, r in enumerate(results) if 
#                             r[2] == "✓" and "negative" not in os.path.basename(test_files[i]).lower())
#         false_positives = sum(1 for i, r in enumerate(results) if 
#                              r[2] == "✓" and "negative" in os.path.basename(test_files[i]).lower())
        
#         if positive_tests > 0:
#             print(f"True positive rate: {true_positives/positive_tests*100:.2f}%")
#         if negative_tests > 0:
#             print(f"False positive rate: {false_positives/negative_tests*100:.2f}%")
        
#         return results


# # Example usage
# if __name__ == "__main__":
#     # This is just a skeleton example showing how to use the class
#     from Detection import ONNXtoTorchModel
#     import librosa

    
#     # Define paths
#     base_dir = "./"
    
#     # Model path
#     model_path = os.path.join(base_dir, "resnet_50_arc", "slim_93%_accuracy_72.7390%.onnx")
#     model = ONNXtoTorchModel(model_path)
    
#     # Audio files
#     positive_files = [
#         os.path.join(base_dir, "tts_samples", "positive", "normal_voice0.wav"),
#         os.path.join(base_dir, "tts_samples", "positive", "normal_voice1.wav"),
#         os.path.join(base_dir, "tts_samples", "positive", "soft_voice0.wav"),
#         os.path.join(base_dir, "tts_samples", "positive", "soft_voice1.wav"),
#         os.path.join(base_dir, "tts_samples", "positive", "clear_voice0.wav"),
#         os.path.join(base_dir, "tts_samples", "positive", "clear_voice1.wav")
#     ]
    
#     negative_files = [
#         os.path.join(base_dir, "tts_samples", "negative", "partial_voice0.wav"),
#         os.path.join(base_dir, "tts_samples", "negative", "partial_voice1.wav"),
#         os.path.join(base_dir, "tts_samples", "negative", "last_part_voice0.wav"),
#         os.path.join(base_dir, "tts_samples", "negative", "last_part_voice1.wav")
#     ]
    
#     test_files = [
#         os.path.join(base_dir, "Recording.wav"),
#         os.path.join(base_dir, "Recording (2).wav"),
#         os.path.join(base_dir, "Recording (3).wav"),
#         os.path.join(base_dir, "Recording (4).wav"),
#         os.path.join(base_dir, "Recording (5).wav"),
#         os.path.join(base_dir, "Recording (6).wav"),
#         os.path.join(base_dir, "Recording (7).wav"),
#         os.path.join(base_dir, "dim_recording.wav"),
#         os.path.join(base_dir, "dim_recording2.wav"),
#         os.path.join(base_dir, "faint_voice.wav"),
#         os.path.join(base_dir, "faint_voice2.wav"),
#         os.path.join(base_dir, "Recording_negative (1).wav"),
#         os.path.join(base_dir, "Recording_negative (2).wav"),
#         os.path.join(base_dir, "Recording_negative (3).wav"),
#         os.path.join(base_dir, "Recording_negative (4).wav"),
#         os.path.join(base_dir, "Recording_negative (5).wav"),
#         os.path.join(base_dir, "dim_recording_negative (1).wav"),
#         os.path.join(base_dir, "dim_recording_negative (2).wav"),
#         os.path.join(base_dir, "dim_recording_negative (3).wav"),
#         os.path.join(base_dir, "dim_recording_negative (4).wav"),
#         os.path.join(base_dir, "dim_recording_negative (5).wav"),
#         os.path.join(base_dir, "dim_recording_negative (6).wav"),
#         os.path.join(base_dir, "dim_recording_negative (7).wav"),
#     ]
        
#     # Process positive examples
#     print(f"{Fore.GREEN}Processing positive examples...{Style.RESET_ALL}")
#     positive_embeddings = []
#     for file in positive_files:
        
#         audio, sr = librosa.load(file, sr=16000)
#         # Ensure audio is exactly 24000 samples long
#         expected_length = 24000
#         if len(audio) < expected_length:
#             pad_length = expected_length - len(audio)
#             audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros
        
#         emb = model(audio)
#         positive_embeddings.append(emb)
    
#     # Process negative examples
#     print(f"{Fore.RED}Processing negative examples...{Style.RESET_ALL}")
#     negative_embeddings = []
#     for file in negative_files:
        
#         audio, sr = librosa.load(file, sr=16000)
#         # Ensure audio is exactly 24000 samples long
#         expected_length = 24000
#         if len(audio) < expected_length:
#             pad_length = expected_length - len(audio)
#             audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros
        
#         emb = model(audio)
#         negative_embeddings.append(emb)
    
#     # Initialize matcher
#     matcher = EnhancedSimilarityMatcher(positive_embeddings, negative_embeddings)
    
#     # Run comprehensive test
#     results = matcher.run_comprehensive_test(model, test_files)



# import pyaudio
# import wave
# import numpy as np
# import os
# import time
# import sounddevice as sd
# import soundfile as sf
# from pathlib import Path
# import requests
# import tempfile
# from typing import List, Dict, Optional
# import json
# from colorama import Fore, Style, init

# # Initialize colorama for cross-platform colored terminal output
# init()

# class WakeWordRecorder:
#     def __init__(self, 
#                  wake_word: str,
#                  sample_rate: int = 16000, 
#                  channels: int = 1,
#                  duration: int = 3,
#                  output_dir: str = "./reference"):
#         self.wake_word = wake_word
#         self.sample_rate = sample_rate
#         self.channels = channels
#         self.duration = duration  # recording duration in seconds
#         self.chunk = 1024
#         self.format = pyaudio.paInt16
        
#         # Ensure output directories exist
#         self.output_dir = Path(output_dir)
#         self.positive_dir = self.output_dir / "positive"
#         self.positive_dir.mkdir(parents=True, exist_ok=True)
        
#         # Audio device setup
#         self.audio = pyaudio.PyAudio()
        
#     def _record_audio(self) -> np.ndarray:
#         """Record audio from microphone"""
#         print(f"{Fore.YELLOW}Recording will start in 3 seconds...{Style.RESET_ALL}")
#         for i in range(3, 0, -1):
#             print(f"{i}...")
#             time.sleep(1)
            
#         print(f"{Fore.GREEN}Recording now! Speak your wake word...{Style.RESET_ALL}")
        
#         frames = []
#         stream = self.audio.open(
#             format=self.format,
#             channels=self.channels,
#             rate=self.sample_rate,
#             input=True,
#             frames_per_buffer=self.chunk
#         )
        
#         # Record for specified duration
#         for _ in range(0, int(self.sample_rate / self.chunk * self.duration)):
#             data = stream.read(self.chunk)
#             frames.append(data)
        
#         print(f"{Fore.GREEN}Recording complete!{Style.RESET_ALL}")
        
#         # Close stream
#         stream.stop_stream()
#         stream.close()
        
#         # Convert to numpy array
#         audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
#         return audio_data
    
#     def _save_audio(self, audio_data: np.ndarray, filename: str) -> str:
#         """Save audio data to a WAV file"""
#         filepath = self.positive_dir / filename
        
#         with wave.open(str(filepath), 'wb') as wf:
#             wf.setnchannels(self.channels)
#             wf.setsampwidth(self.audio.get_sample_size(self.format))
#             wf.setframerate(self.sample_rate)
#             wf.writeframes(audio_data.tobytes())
        
#         return str(filepath)
    
#     def _play_audio(self, filepath: str) -> None:
#         """Play audio file for confirmation"""
#         print(f"{Fore.BLUE}Playing back your recording...{Style.RESET_ALL}")
#         data, fs = sf.read(filepath)
#         sd.play(data, fs)
#         sd.wait()
    
#     def generate_tts_samples(self, api_key: Optional[str] = None) -> List[str]:
#         """Generate TTS samples of the wake word using an AI API"""
#         if not api_key:
#             print(f"{Fore.RED}No API key provided. Skipping TTS generation.{Style.RESET_ALL}")
#             return []
        
#         variations = [
#             f"{self.wake_word}",
#             f"{self.wake_word}?",
#             f"Hey, {self.wake_word}",
#             f"{self.wake_word}, are you there?",
#             f"Um, {self.wake_word}",
#             f"{self.wake_word}, I need your help"
#         ]
        
#         tts_files = []
        
#         try:
#             # Here we'd implement the API call to a TTS service
#             # For example, using OpenAI's TTS API or ElevenLabs
#             # This is a placeholder for the actual implementation
            
#             print(f"{Fore.BLUE}Generating AI voice samples for '{self.wake_word}'...{Style.RESET_ALL}")
            
#             # Placeholder for API call
#             for i, text in enumerate(variations):
#                 # In a real implementation, this would make an API call
#                 # Here we're just creating placeholder files
#                 filepath = self.positive_dir / f"tts_sample_{i}.wav"
                
#                 # Placeholder: In reality, we'd save the API response
#                 with open(filepath, 'wb') as f:
#                     # Generate a short silence file as placeholder
#                     temp_audio = np.zeros(self.sample_rate * 2, dtype=np.int16)
#                     sf.write(filepath, temp_audio, self.sample_rate)
                
#                 tts_files.append(str(filepath))
#                 print(f"  Generated sample for: '{text}'")
            
#             print(f"{Fore.GREEN}Successfully generated {len(tts_files)} TTS samples{Style.RESET_ALL}")
            
#         except Exception as e:
#             print(f"{Fore.RED}Error generating TTS samples: {e}{Style.RESET_ALL}")
        
#         return tts_files
    
#     def record_samples(self) -> Dict[str, List[str]]:
#         """Record different variations of the wake word"""
#         recording_types = [
#             {"name": "normal", "prompt": f"Say the wake word '{self.wake_word}' normally or casually"},
#             {"name": "quick", "prompt": f"Say the wake word '{self.wake_word}' quickly"},
#             {"name": "loud", "prompt": f"Shout the wake word '{self.wake_word}'"},
#             {"name": "whisper", "prompt": f"Whisper the wake word '{self.wake_word}'"}
#         ]
        
#         recordings = {"user_samples": [], "ai_samples": []}
        
#         # Record user samples
#         print(f"{Fore.CYAN}===== RECORDING WAKE WORD SAMPLES FOR '{self.wake_word}' ====={Style.RESET_ALL}")
#         print("We'll record 4 different variations of how you say your wake word.")
#         print("This helps create a more robust model that responds to different speaking styles.")
        
#         for idx, rec_type in enumerate(recording_types):
#             print(f"\n{Fore.CYAN}Recording {idx+1}/4: {rec_type['name'].upper()}{Style.RESET_ALL}")
#             input(f"{rec_type['prompt']}. Press Enter to start recording...")
            
#             # Record audio
#             audio_data = self._record_audio()
            
#             # Save to file
#             filename = f"{self.wake_word.replace(' ', '_')}_{rec_type['name']}.wav"
#             filepath = self._save_audio(audio_data, filename)
#             recordings["user_samples"].append(filepath)
            
#             # Play back for confirmation
#             self._play_audio(filepath)
            
#             while True:
#                 response = input(f"Is this recording good? (y/n): ").lower()
#                 if response == 'y':
#                     print(f"{Fore.GREEN}Great! Moving to next recording.{Style.RESET_ALL}")
#                     break
#                 elif response == 'n':
#                     print(f"{Fore.YELLOW}Let's try again.{Style.RESET_ALL}")
#                     audio_data = self._record_audio()
#                     filepath = self._save_audio(audio_data, filename)
#                     recordings["user_samples"][-1] = filepath
#                     self._play_audio(filepath)
#                 else:
#                     print(f"{Fore.RED}Please enter 'y' or 'n'.{Style.RESET_ALL}")
        
#         # Ask if user wants AI-generated samples as well
#         print(f"\n{Fore.CYAN}Would you like to generate AI voice samples as additional training data?{Style.RESET_ALL}")
#         print("This can improve detection accuracy with varied speech patterns.")
        
#         response = input("Generate AI samples? (y/n): ").lower()
#         if response == 'y':
#             api_key = input("Enter your TTS API key (leave blank to skip): ").strip()
#             if api_key:
#                 ai_samples = self.generate_tts_samples(api_key)
#                 recordings["ai_samples"] = ai_samples
#             else:
#                 print(f"{Fore.YELLOW}No API key provided. Skipping AI sample generation.{Style.RESET_ALL}")
        
#         # Cleanup
#         self.audio.terminate()
        
#         # Save metadata
#         metadata = {
#             "wake_word": self.wake_word,
#             "sample_rate": self.sample_rate,
#             "channels": self.channels,
#             "user_samples": recordings["user_samples"],
#             "ai_samples": recordings["ai_samples"],
#             "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
#         }
        
#         with open(self.output_dir / "metadata.json", "w") as f:
#             json.dump(metadata, f, indent=4)
        
#         return recordings

# def create_embeddings(recordings: Dict[str, List[str]], model_path: str) -> None:
#     """Create embeddings from recorded samples"""
#     # This would use your ONNXtoTorchModel to create embeddings
#     # For the recorded samples and save them to a reference file
    
#     print(f"{Fore.CYAN}Generating embeddings from recorded samples...{Style.RESET_ALL}")
    
#     # Placeholder for the actual embedding creation
#     # In a real implementation, this would:
#     # 1. Load your ONNX model
#     # 2. Process each audio file to generate embeddings
#     # 3. Save the embeddings to a reference file
    
#     print(f"{Fore.GREEN}Successfully created embeddings for wake word detection!{Style.RESET_ALL}")

# def main():
#     print(f"{Fore.CYAN}===== WAKE WORD RECORDER =====\n{Style.RESET_ALL}")
    
#     wake_word = input("What wake word would you like to use? (e.g., 'Hey Assistant'): ")
    
#     if not wake_word.strip():
#         print(f"{Fore.RED}Error: Wake word cannot be empty.{Style.RESET_ALL}")
#         return
    
#     # Create recorder instance
#     recorder = WakeWordRecorder(wake_word=wake_word)
    
#     # Record samples
#     recordings = recorder.record_samples()
    
#     # Create embeddings from recordings
#     model_path = "./resnet_50_arc/slim_93%_accuracy_72.7390%.onnx"
#     create_embeddings(recordings, model_path)
    
#     print(f"\n{Fore.GREEN}===== WAKE WORD SETUP COMPLETE ====={Style.RESET_ALL}")
#     print(f"Wake word '{wake_word}' has been set up with {len(recordings['user_samples'])} user samples " +
#           f"and {len(recordings['ai_samples'])} AI samples.")
#     print(f"Reference files saved to: {recorder.output_dir}")
#     print(f"\n{Fore.YELLOW}You can now use this wake word with your voice assistant.{Style.RESET_ALL}")

# if __name__ == "__main__":
#     main()


# if __name__ == "__main__":
#     # Define paths
#     base_dir = "./"
    
#     # Model path
#     model_path = os.path.join(base_dir, "resnet_50_arc", "slim_93%_accuracy_72.7390%.onnx")
    
#     # Audio files
#     positive_files = [
#         os.path.join(base_dir, "tts_samples", "positive", "normal_voice0.wav"),
#         os.path.join(base_dir, "tts_samples", "positive", "normal_voice1.wav"),
#         os.path.join(base_dir, "tts_samples", "positive", "soft_voice0.wav"),
#         os.path.join(base_dir, "tts_samples", "positive", "soft_voice1.wav"),
#         os.path.join(base_dir, "tts_samples", "positive", "clear_voice0.wav"),
#         os.path.join(base_dir, "tts_samples", "positive", "clear_voice1.wav")
#     ]
    
#     negative_files = [
#         os.path.join(base_dir, "tts_samples", "negative", "partial_voice0.wav"),
#         os.path.join(base_dir, "tts_samples", "negative", "partial_voice1.wav"),
#         os.path.join(base_dir, "tts_samples", "negative", "last_part_voice0.wav"),
#         os.path.join(base_dir, "tts_samples", "negative", "last_part_voice1.wav")
#     ]
    
#     test_files = [
#         os.path.join(base_dir, "Recording.wav"),
#         os.path.join(base_dir, "Recording (2).wav"),
#         os.path.join(base_dir, "Recording (3).wav"),
#         os.path.join(base_dir, "Recording (4).wav"),
#         os.path.join(base_dir, "Recording (5).wav"),
#         os.path.join(base_dir, "Recording (6).wav"),
#         os.path.join(base_dir, "Recording (7).wav"),
#         os.path.join(base_dir, "dim_recording.wav"),
#         os.path.join(base_dir, "dim_recording2.wav"),
#         os.path.join(base_dir, "faint_voice.wav"),
#         os.path.join(base_dir, "faint_voice2.wav")
#     ]
    
#     # Additional test with noisy samples
#     # test_files.extend([
#     #     os.path.join(base_dir, "tts_samples", "positive", "normal_voice0_noisy.wav"),
#     #     os.path.join(base_dir, "tts_samples", "positive", "soft_voice0_noisy.wav")
#     # ])
    
#     # Run tests
#     tester = WakeWordTester(model_path)
#     results = tester.run_comprehensive_test(positive_files, negative_files, test_files)





# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy.stats import norm
# import torch
# import torch.nn.functional as F
# from Detection import ONNXtoTorchModel

# class EnhancedSimilarityMatcher:
#     def __init__(self, positive_embeddings, negative_embeddings=None, noise_levels=None):
#         """
#         Initialize with positive and optional negative example embeddings
#         """
#         # Convert embeddings to numpy and ensure correct shape
#         self.positive_embeddings = np.array([emb.squeeze().cpu().numpy() for emb in positive_embeddings])
#         self.negative_embeddings = np.array([emb.squeeze().cpu().numpy() for emb in negative_embeddings]) if negative_embeddings else None
#         self.noise_levels = np.array(noise_levels) if noise_levels else None
        
#         # Calculate statistics from positive examples
#         self.positive_centroid = np.mean(self.positive_embeddings, axis=0)
#         self.positive_std = np.std(self.positive_embeddings, axis=0)
        
#         # Calculate decision boundaries if negative samples exist
#         if self.negative_embeddings is not None:
#             self.negative_centroid = np.mean(self.negative_embeddings, axis=0)
#             self._calculate_decision_boundary()
    
#     def _calculate_decision_boundary(self):
#         """Calculate optimal decision boundary using positive and negative samples"""
#         pos_sims = self._batch_cosine_similarity(self.positive_embeddings, self.positive_centroid)
#         neg_sims = self._batch_cosine_similarity(self.negative_embeddings, self.positive_centroid)
        
#         # Find optimal threshold that maximizes separation
#         all_sims = np.concatenate([pos_sims, neg_sims])
#         all_sims.sort()
        
#         best_threshold = 0
#         best_separation = -float('inf')
        
#         for threshold in all_sims:
#             pos_correct = np.mean(pos_sims >= threshold)
#             neg_correct = np.mean(neg_sims < threshold)
#             separation = pos_correct + neg_correct - 1
#             if separation > best_separation:
#                 best_separation = separation
#                 best_threshold = threshold
        
#         self.decision_threshold = best_threshold
    
#     def _batch_cosine_similarity(self, embeddings, reference):
#         """Calculate cosine similarity between embeddings and reference"""
#         # Ensure shapes are correct
#         if len(embeddings.shape) == 3:
#             embeddings = embeddings.squeeze(1)
#         if len(reference.shape) == 1:
#             reference = reference.reshape(1, -1)
            
#         return cosine_similarity(embeddings, reference).flatten()
    
#     def _adaptive_gaussian_kernel(self, distance, noise_level=0):
#         """Compute Gaussian kernel with adaptive sigma based on noise level"""
#         base_sigma = 0.4  # Increased to be more lenient with variations
#         max_sigma = 0.6  # Increased accordingly
        
#         adaptive_sigma = base_sigma + (max_sigma - base_sigma) * noise_level
#         return norm.pdf(distance, loc=0, scale=adaptive_sigma)
    
#     def compute_enhanced_similarity(self, query_embedding, noise_level=0):
#         """Compute enhanced similarity score using multiple metrics"""
#         # Ensure query_embedding is the right shape
#         query_embedding = query_embedding.squeeze()
        
#         # 1. Cosine similarity with positive centroid
#         cosine_sim = cosine_similarity(
#             query_embedding.reshape(1, -1), 
#             self.positive_centroid.reshape(1, -1)
#         )[0][0]
        
#         # 2. Average similarity to positive examples
#         pos_similarities = self._batch_cosine_similarity(self.positive_embeddings, query_embedding)
#         avg_pos_sim = np.mean(pos_similarities)
        
#         # 3. Distance from negative samples (if available)
#         negative_penalty = 0
#         if self.negative_embeddings is not None:
#             neg_sims = self._batch_cosine_similarity(self.negative_embeddings, query_embedding)
#             negative_penalty = np.mean(neg_sims)
        
#         # 4. Gaussian kernel similarity with adaptive sigma
#         embedding_distance = np.linalg.norm(query_embedding - self.positive_centroid)
#         gaussian_sim = self._adaptive_gaussian_kernel(embedding_distance, noise_level)
        
#         # 5. Standard deviation check (penalize outliers)
#         std_penalty = np.mean(
#             np.abs(query_embedding - self.positive_centroid) > 2 * self.positive_std
#         )
        
#         # Adjusted weights with more emphasis on positive similarities
#         weights = {
#             'cosine': 0.40,    # Increased to give more importance to direct similarity
#             'avg_pos': 0.30,   # Increased to better handle new positive samples
#             'gaussian': 0.20,  # Slightly reduced
#             'negative': 0.15,  # Increased to better penalize negative samples
#             'std': 0.05       # Kept same
#         }
        
#         # Adjust weights based on noise level
#         if noise_level > 0.5:
#             weights['gaussian'] += 0.05  # Reduced from 0.1
#             weights['cosine'] -= 0.025   # Reduced penalty
#             weights['avg_pos'] -= 0.025  # Reduced penalty
        
#         final_score = (
#             weights['cosine'] * cosine_sim +
#             weights['avg_pos'] * avg_pos_sim +
#             weights['gaussian'] * gaussian_sim -
#             weights['negative'] * negative_penalty -
#             weights['std'] * std_penalty
#         )
        
#         # Normalize score to [0, 1] range
#         final_score = (final_score + 1) / 2
        
#         return np.clip(final_score, 0, 1)  # Ensure score is between 0 and 1
    
#     def is_wake_word(self, query_embedding, noise_level=0, threshold=None):
#         """Determine if the query embedding represents the wake word"""
#         similarity = self.compute_enhanced_similarity(query_embedding, noise_level)
        
#         if threshold is None:
#             # Change to
#             # threshold = self.decision_threshold if hasattr(self, 'decision_threshold') else 0.55  # Lowered default threshold
#             threshold = 0.55
            
#         return similarity > threshold, similarity

# def estimate_noise_level(audio, sr=16000):
#     """Estimate noise level in audio signal"""
#     signal_power = np.mean(audio ** 2)
#     peak_power = np.max(audio ** 2)
    
#     if peak_power > 0:
#         snr = 10 * np.log10(peak_power / signal_power)
#         noise_level = 1 / (1 + np.exp(0.1 * (snr - 10)))
#     else:
#         noise_level = 1.0
        
#     return np.clip(noise_level, 0, 1)

# if __name__ == "__main__":
    
    # import librosa
    
    # model = ONNXtoTorchModel(r"C:\Users\Rohit Francis\Desktop\Codes\Pathor Wake Word\EfficientWord-Net\eff_word_net\models\resnet_50_arc\slim_93%_accuracy_72.7390%.onnx")
    # # audio = np.random.randn(24000)  # 1.5 seconds at 16kHz
    # file_path = "audio2.wav"
    # # Load and resample to 16 kHz
    # audio, sr = librosa.load(file_path, sr=16000)
    # # Ensure audio is exactly 24000 samples long
    # expected_length = 24000
    # if len(audio) < expected_length:
    #     pad_length = expected_length - len(audio)
    #     audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros

    # print("Audio processing: ", audio.shape, sr)
    # embeddings_audio = model(audio)
    # print("Embeddings from audio shape:", embeddings_audio.shape)
    
    # # audio = np.random.randn(24000)  # 1.5 seconds at 16kHz
    # file_path2 = "audio2_twin.wav"
    # # Load and resample to 16 kHz
    # audio2, sr = librosa.load(file_path2, sr=16000)
    # # Ensure audio is exactly 24000 samples long
    # expected_length = 24000
    # if len(audio2) < expected_length:
    #     pad_length = expected_length - len(audio2)
    #     audio2 = np.pad(audio2, (0, pad_length), mode='constant')  # Pad with zeros

    # embeddings_audio2 = model(audio2)
         
    
#     file_path = "audio.wav"
#     # Load and resample to 16 kHz
#     audio, sr = librosa.load(file_path, sr=16000)
#     # Ensure audio is exactly 24000 samples long
#     expected_length = 24000
#     if len(audio) < expected_length:
#         pad_length = expected_length - len(audio)
#         audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros

#     print("Audio processing: ", audio.shape, sr)
#     embeddings_audio3 = model(audio)
#     print("Embeddings from audio shape:", embeddings_audio3.shape)
    
    
#     file_path = "audio_twin.wav"
#     # Load and resample to 16 kHz
#     audio, sr = librosa.load(file_path, sr=16000)
#     # Ensure audio is exactly 24000 samples long
#     expected_length = 24000
#     if len(audio) < expected_length:
#         pad_length = expected_length - len(audio)
#         audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros

#     print("Audio processing: ", audio.shape, sr)
#     embeddings_audio4 = model(audio)
#     print("Embeddings from audio shape:", embeddings_audio4.shape)
    
#     positive_embeddings = [embeddings_audio, embeddings_audio2]
#     negative_embeddings = [embeddings_audio3, embeddings_audio4]
    
    
    
    
#     file_path = "test.wav"
#     # Load and resample to 16 kHz
#     audio, sr = librosa.load(file_path, sr=16000)
#     # Ensure audio is exactly 24000 samples long
#     expected_length = 24000
#     if len(audio) < expected_length:
#         pad_length = expected_length - len(audio)
#         audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros

#     query_embeddings = model(audio)
#     query_embeddings = query_embeddings.detach().numpy()
#     noise_levels = estimate_noise_level(audio)
    
#     matcher = EnhancedSimilarityMatcher(positive_embeddings, negative_embeddings, noise_levels)    

#     # For detection
#     noise_level = estimate_noise_level(audio)
#     is_wake_word, confidence = matcher.is_wake_word(query_embeddings, noise_level)

#     print(is_wake_word, confidence)



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

