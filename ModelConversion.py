import tensorflow as tf
import numpy as np
import tf2onnx
import onnx
import os
from Detection import ONNXtoTorchModel

# First recreate the cosine_similarity function from your code
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

# Recreate the EnhancedSimilarityMatcher class
class EnhancedSimilarityMatcher(tf.keras.Model):
    def __init__(self, positive_embeddings, negative_embeddings=None, noise_levels=None):
        super().__init__()
        
        self.positive_embeddings = tf.constant(positive_embeddings)
        self.negative_embeddings = tf.constant(negative_embeddings) if negative_embeddings is not None else None
        self.noise_levels = tf.constant(noise_levels) if noise_levels else None
        
        # Calculate statistics from positive examples
        self.positive_centroid = tf.reduce_mean(self.positive_embeddings, axis=0)
        self.positive_std = tf.math.reduce_std(self.positive_embeddings, axis=0)
        
        if self.negative_embeddings is not None:
            if len(tf.shape(self.negative_embeddings)) > 1:
                self.negative_centroid = tf.reduce_mean(self.negative_embeddings, axis=0)
            else:
                self.negative_centroid = self.negative_embeddings
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
        
    def _batch_cosine_similarity(self, embeddings, reference):
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


def convert_model_to_onnx():
    # Create sample data for model initialization
    # Assuming embeddings are 128-dimensional vectors
    embedding_dim = 2048
    
    import librosa
    from colorama import Fore, Style
    import tf2onnx
    
    base_dir = "./"
    
    model_path = os.path.join(base_dir, "resnet_50_arc", "slim_93%_accuracy_72.7390%.onnx")
    model = ONNXtoTorchModel(model_path)
    
    
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
        positive_embeddings.append(emb.detach().numpy())
    
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
        negative_embeddings.append(emb.detach().numpy())
    
    # Initialize matcher
    similarity_model = EnhancedSimilarityMatcher(positive_embeddings, negative_embeddings)
    
    # Define the model call function with tf.function decorator
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, embedding_dim], dtype=tf.float32, name="query_embedding"),
        tf.TensorSpec(shape=[], dtype=tf.float32, name="noise_level")
    ])
    def model_call(query_embedding, noise_level):
        return similarity_model.call(query_embedding, noise_level)
    
    # Define output path for ONNX model
    output_path = "similarity_matcher.onnx"
    
    # Convert the model call function to ONNX format
    model_proto, _ = tf2onnx.convert.from_function(
        model_call,
        input_signature=[
            tf.TensorSpec(shape=[None, embedding_dim], dtype=tf.float32, name="query_embedding"),
            tf.TensorSpec(shape=[], dtype=tf.float32, name="noise_level")
        ],
        opset=13,
        output_path=output_path
    )
    
    print(f"Model converted and saved to {output_path}")
    
    # Create wrapper function for noise level estimation
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32, name="audio")
    ])
    def estimate_noise_wrapper(audio):
        return similarity_model.estimate_noise_level(audio)
    
    noise_level_path = "noise_level_estimator.onnx"
    noise_model_proto, _ = tf2onnx.convert.from_function(
        estimate_noise_wrapper,
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32, name="audio")],
        opset=13,
        output_path=noise_level_path
    )
    
    print(f"Noise estimator converted and saved to {noise_level_path}")
    
    # Create a function for similarity score only
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, embedding_dim], dtype=tf.float32, name="query_embedding"),
        tf.TensorSpec(shape=[], dtype=tf.float32, name="noise_level")
    ])
    def get_similarity_only(query_embedding, noise_level):
        similarity, _ = similarity_model.compute_enhanced_similarity(query_embedding, noise_level)
        return similarity
    
    similarity_only_path = "similarity_score.onnx"
    similarity_model_proto, _ = tf2onnx.convert.from_function(
        get_similarity_only,
        input_signature=[
            tf.TensorSpec(shape=[None, embedding_dim], dtype=tf.float32, name="query_embedding"),
            tf.TensorSpec(shape=[], dtype=tf.float32, name="noise_level")
        ],
        opset=13,
        output_path=similarity_only_path
    )
    
    print(f"Similarity calculator converted and saved to {similarity_only_path}")
    
    # Create a function for is_wake_word prediction
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, embedding_dim], dtype=tf.float32, name="query_embedding"),
        tf.TensorSpec(shape=[], dtype=tf.float32, name="noise_level"),
        tf.TensorSpec(shape=[], dtype=tf.float32, name="threshold")  # Removed default parameter
    ])
    def is_wake_word_function(query_embedding, noise_level, threshold):
        # Use the threshold parameter directly without default
        is_wake, similarity, _ = similarity_model.is_wake_word(query_embedding, noise_level, threshold)
        return {"is_wake": is_wake, "similarity": similarity}

    wake_word_path = "wake_word_detector.onnx"
    wake_word_proto, _ = tf2onnx.convert.from_function(
        is_wake_word_function,
        input_signature=[
            tf.TensorSpec(shape=[None, embedding_dim], dtype=tf.float32, name="query_embedding"),
            tf.TensorSpec(shape=[], dtype=tf.float32, name="noise_level"),
            tf.TensorSpec(shape=[], dtype=tf.float32, name="threshold")  # Removed default parameter
        ],
        opset=13,
        output_path=wake_word_path
    )

    print(f"Wake word detector converted and saved to {wake_word_path}")
    
    return output_path, noise_level_path, similarity_only_path, wake_word_path

if __name__ == "__main__":
    # Check if TensorFlow version is compatible with tf2onnx
    print(f"TensorFlow version: {tf.__version__}")
    
    try:
        # Convert the model
        model_path, noise_path, sim_path = convert_model_to_onnx()
        
        # Verify the ONNX models
        for path in [model_path, noise_path, sim_path]:
            try:
                # Load and check the model
                onnx_model = onnx.load(path)
                onnx.checker.check_model(onnx_model)
                print(f"ONNX model {path} verified successfully!")
            except Exception as e:
                print(f"Error verifying ONNX model {path}: {e}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")