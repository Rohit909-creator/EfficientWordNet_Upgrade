import os
import numpy as np
import librosa
import glob
import torch
from Detection import ONNXtoTorchModel
from sklearn.metrics.pairwise import cosine_similarity
from colorama import Fore, Style
import argparse

def find_wake_word_files(base_dir):
    """Find all wake word audio files in the base directory."""
    wake_word_files = []
    
    # Path to recordings directory
    recordings_dir = os.path.join(base_dir, "wake_word_data", "recordings")
    
    # Search types (subdirectories)
    types = ["normal", "quick", "shouted", "whispered"]
    
    for type_dir in types:
        search_path = os.path.join(recordings_dir, type_dir, "*.wav")
        files = glob.glob(search_path)
        wake_word_files.extend(files)
    
    # Extract unique wake words from filenames
    wake_words = set()
    for file_path in wake_word_files:
        filename = os.path.basename(file_path)
        # Extract wake word from filename (format: WakeWord_type_num.wav)
        wake_word = filename.split('_')[0]
        wake_words.add(wake_word)
    
    return list(wake_words), wake_word_files

def get_wake_word_embeddings(model, file_path, expected_length=24000):
    """Generate embeddings for a single audio file"""
    print(f"Processing: {os.path.basename(file_path)}")
    
    # Load and resample audio
    audio, sr = librosa.load(file_path, sr=16000)
    
    # Ensure audio is exactly the expected length
    if len(audio) < expected_length:
        audio = np.pad(audio, (0, expected_length - len(audio)), mode='constant')
    audio = audio[:expected_length]
    
    # Generate embedding
    embedding = model(audio)
    
    return embedding, audio

def estimate_noise_level(audio):
    """Estimate noise level in audio signal"""
    signal_power = np.mean(audio ** 2)
    peak_power = np.max(audio ** 2)
    
    if peak_power > 0:
        snr = 10 * np.log10(peak_power / signal_power)
        noise_level = 1 / (1 + np.exp(0.1 * (snr - 10)))
    else:
        noise_level = 1.0
    return np.clip(noise_level, 0, 1)

def calculate_adaptive_similarity(target_emb, word_emb, target_audio, word_audio):
    """
    Calculate similarity between embeddings with noise adaptation
    
    Args:
        target_emb: Embedding of target wake word
        word_emb: Embedding of comparison wake word
        target_audio: Audio of target wake word for noise estimation
        word_audio: Audio of comparison wake word for noise estimation
    """
    # Extract raw embeddings
    if isinstance(target_emb, torch.Tensor):
        target_emb = target_emb.detach().cpu().numpy()
    if isinstance(word_emb, torch.Tensor):
        word_emb = word_emb.detach().cpu().numpy()
    
    # Reshape if needed
    if len(target_emb.shape) > 1:
        target_emb = target_emb.squeeze()
    if len(word_emb.shape) > 1:
        word_emb = word_emb.squeeze()
        
    # Calculate standard cosine similarity
    cosine_sim = cosine_similarity(target_emb.reshape(1, -1), word_emb.reshape(1, -1))[0][0]
    
    # Estimate noise levels
    target_noise = estimate_noise_level(target_audio)
    word_noise = estimate_noise_level(word_audio)
    
    # Get maximum noise level
    noise_level = max(target_noise, word_noise)
    
    # Calculate enhanced similarity - more sophisticated comparison
    # 1. Gaussian kernel similarity (more sensitive to small differences)
    embedding_distance = np.linalg.norm(target_emb - word_emb)
    base_sigma = 0.4
    max_sigma = 0.6
    adaptive_sigma = base_sigma + (max_sigma - base_sigma) * noise_level
    gaussian_sim = np.exp(-embedding_distance**2 / (2 * adaptive_sigma**2))
    
    # 2. Adjust similarity based on noise level
    # - Higher noise means we should be more lenient (higher similarity scores)
    # - This helps account for how noise affects similarity in real-world conditions
    adjusted_sim = cosine_sim * (1 + 0.2 * noise_level)
    
    # 3. Combined similarity score
    # - Give more weight to cosine similarity in low noise
    # - Give more weight to Gaussian similarity in high noise
    combined_sim = (1 - noise_level) * adjusted_sim + noise_level * gaussian_sim
    
    # Convert to 0-1 scale
    scaled_sim = (combined_sim + 1) / 2 if combined_sim <= 1 else 1.0
    
    similarity_data = {
        'cosine_sim': (cosine_sim + 1) / 2,
        'gaussian_sim': gaussian_sim,
        'noise_level': noise_level,
        'combined_sim': scaled_sim
    }
    
    return scaled_sim, similarity_data

def search_wake_words_embeddings(base_dir, target_word, model_path, 
                                num_similar=3, files_per_group=2):
    """
    Search for wake words similar to the target word using embedding similarity.
    
    Args:
        base_dir: Base directory containing wake word data
        target_word: Target wake word to find similar words to
        model_path: Path to the ONNX model file
        num_similar: Number of similar words to return
        files_per_group: Number of files to select per similar word
    
    Returns:
        Dictionary with search results
    """
    print(f"\n{Fore.CYAN}🔍 Searching for wake words similar to '{target_word}' using embeddings with noise adaptation{Style.RESET_ALL}")
    
    # Initialize model
    model = ONNXtoTorchModel(model_path)
    
    # Find all available wake words and files
    all_wake_words, all_files = find_wake_word_files(base_dir)
    print(f"Found {len(all_wake_words)} unique wake words in the dataset")
    
    # Find target word files
    target_files = [f for f in all_files if os.path.basename(f).startswith(f"{target_word}_")]
    
    if not target_files:
        print(f"{Fore.RED}No files found for target word '{target_word}'{Style.RESET_ALL}")
        return None
    
    # Get embeddings for target word files
    print(f"{Fore.GREEN}Generating embeddings for target word '{target_word}'...{Style.RESET_ALL}")
    target_embeddings = []
    target_audios = []
    for file in target_files:
        embedding, audio = get_wake_word_embeddings(model, file)
        target_embeddings.append(embedding.detach().cpu().numpy())
        target_audios.append(audio)
    
    # Average the embeddings to get a representative embedding for the target word
    target_embedding = np.mean(target_embeddings, axis=0)
    target_audio = target_audios[0]  # Use first audio for noise analysis
    
    # Calculate similarity to other wake words
    print(f"{Fore.YELLOW}Calculating adaptive similarities to other wake words...{Style.RESET_ALL}")
    similarities = []
    similarity_details = {}
    
    for word in all_wake_words:
        if word.lower() == target_word.lower():
            continue
        
        # Get files for this wake word
        word_files = [f for f in all_files if os.path.basename(f).startswith(f"{word}_")]
        
        if not word_files:
            continue
            
        # Limit number of files to process for efficiency
        sample_files = word_files[:min(3, len(word_files))]
        
        # Get embeddings for these files
        word_embeddings = []
        word_audios = []
        for file in sample_files:
            embedding, audio = get_wake_word_embeddings(model, file)
            word_embeddings.append(embedding.detach().cpu().numpy())
            word_audios.append(audio)
        
        # Average the embeddings
        word_embedding = np.mean(word_embeddings, axis=0)
        word_audio = word_audios[0]  # Use first audio for noise analysis
        
        # Calculate adaptive similarity
        similarity, details = calculate_adaptive_similarity(
            target_embedding, word_embedding, target_audio, word_audio
        )
        
        # Store similarity and metadata
        similarities.append((word, similarity))
        similarity_details[word] = details
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Use more conservative thresholds for high similarity group
    # In the original, high similarity was just the top N
    # Now we'll use a dynamic threshold approach
    similarity_values = [s for _, s in similarities]
    if similarity_values:
        # Use percentile-based thresholds instead of fixed thresholds
        # This adapts to the distribution of similarities in your dataset
        high_threshold = np.percentile(similarity_values, 70)  # More conservative (was 75)
        med_threshold = np.percentile(similarity_values, 40)   # More conservative (was 50)
        
        # Group by thresholds
        high_similarity = [(w, s) for w, s in similarities if s >= high_threshold]
        medium_similarity = [(w, s) for w, s in similarities 
                            if s >= med_threshold and s < high_threshold]
        low_similarity = [(w, s) for w, s in similarities if s < med_threshold]
        
        # Limit group sizes
        high_similarity = high_similarity[:num_similar*2]  # Double the size for more coverage
        medium_similarity = medium_similarity[:num_similar]
        low_similarity = low_similarity[-num_similar:]  # Take the least similar
    else:
        high_similarity = []
        medium_similarity = []
        low_similarity = []
    
    # Print results
    print(f"\n{Fore.CYAN}📊 Similarity Analysis (Embedding-Based with Noise Adaptation):{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}High Similarity:{Style.RESET_ALL}")
    for word, score in high_similarity:
        noise = similarity_details[word]['noise_level']
        cosine = similarity_details[word]['cosine_sim']
        print(f"  • {word} (score: {score:.2f}, noise: {noise:.2f}, cosine: {cosine:.2f})")
    
    if medium_similarity:
        print(f"\n{Fore.YELLOW}Neutral Similarity:{Style.RESET_ALL}")
        for word, score in medium_similarity:
            noise = similarity_details[word]['noise_level']
            cosine = similarity_details[word]['cosine_sim']
            print(f"  • {word} (score: {score:.2f}, noise: {noise:.2f}, cosine: {cosine:.2f})")
    
    if low_similarity:
        print(f"\n{Fore.RED}Low Similarity:{Style.RESET_ALL}")
        for word, score in low_similarity:
            noise = similarity_details[word]['noise_level']
            cosine = similarity_details[word]['cosine_sim']
            print(f"  • {word} (score: {score:.2f}, noise: {noise:.2f}, cosine: {cosine:.2f})")
    
    # Prepare similarity groups for result
    similarity_groups = {
        "high_similarity": [word for word, _ in high_similarity],
        "neutral_similarity": [word for word, _ in medium_similarity],
        "low_similarity": [word for word, _ in low_similarity]
    }
    
    # Get positive files (target word recordings)
    positive_files = target_files
    
    # Get negative files from similarity groups
    negative_files = []
    
    # Add files from high similarity words
    for word, _ in high_similarity:
        word_files = [f for f in all_files if os.path.basename(f).startswith(f"{word}_")]
        
        # Try to get a diverse set of speaking styles
        styles = {}
        for file in word_files:
            filename = os.path.basename(file)
            style = filename.split('_')[1]  # normal, quick, shouted, whispered
            if style not in styles and len(styles) < files_per_group:
                styles[style] = file
        
        # If we couldn't get diverse styles, fall back to first N files
        if len(styles) < files_per_group:
            selected_files = word_files[:files_per_group]
        else:
            selected_files = list(styles.values())
            
        negative_files.extend(selected_files)
    
    # Add files from medium similarity words
    for word, _ in medium_similarity:
        word_files = [f for f in all_files if os.path.basename(f).startswith(f"{word}_")]
        sample_size = min(files_per_group, len(word_files))
        negative_files.extend(word_files[:sample_size])
    
    # Add files from low similarity words
    for word, _ in low_similarity:
        word_files = [f for f in all_files if os.path.basename(f).startswith(f"{word}_")]
        sample_size = min(files_per_group, len(word_files))
        negative_files.extend(word_files[:sample_size])
    
    # Print file lists
    print(f"\n{Fore.GREEN}✅ Positive Files:{Style.RESET_ALL}")
    for file in positive_files[:5]:  # Show first 5 for brevity
        print(f"  • {os.path.basename(file)}")
    
    print(f"\n{Fore.RED}❌ Negative Files (selected based on embedding similarity):{Style.RESET_ALL}")
    for file in negative_files[:10]:  # Show first 10 for brevity
        # Determine which group this file belongs to
        filename = os.path.basename(file)
        word = filename.split('_')[0]
        
        group = "unknown"
        if word in similarity_groups["high_similarity"]:
            group = "high_similarity"
        elif word in similarity_groups["neutral_similarity"]:
            group = "neutral_similarity"
        elif word in similarity_groups["low_similarity"]:
            group = "low_similarity"
            
        print(f"  • {filename} ({group})")
    
    # Return results dictionary
    return {
        "target_word": target_word,
        "similarity_groups": similarity_groups,
        "similarity_scores": {
            "high_similarity": high_similarity,
            "neutral_similarity": medium_similarity,
            "low_similarity": low_similarity
        },
        "similarity_details": similarity_details,
        "positive_files": positive_files,
        "negative_files": negative_files
    }

def test_multiple_wake_words(base_dir, model_path, target_words, num_similar=3, files_per_group=2):
    """Test multiple wake words at once."""
    for target_word in target_words:
        print(f"\n{'='*60}")
        search_result = search_wake_words_embeddings(
            base_dir, 
            target_word, 
            model_path,
            num_similar=num_similar,
            files_per_group=files_per_group
        )
        print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="Search for similar wake words using embeddings with noise adaptation")
    parser.add_argument("--base_dir", default=".", help="Base directory containing wake word data")
    parser.add_argument("--model_path", default="./resnet_50_arc/slim_93%_accuracy_72.7390%.onnx", 
                      help="Path to the ONNX model file")
    parser.add_argument("--wake_words", nargs="+", default=["Alexa", "Siri", "Ava"],
                      help="List of wake words to search for")
    parser.add_argument("--num_similar", type=int, default=3, 
                      help="Number of similar words to find for each group")
    parser.add_argument("--files_per_group", type=int, default=2,
                      help="Number of files to select per similar word")
    
    args = parser.parse_args()
    
    # Test multiple wake words
    test_multiple_wake_words(
        args.base_dir,
        args.model_path,
        args.wake_words,
        args.num_similar,
        args.files_per_group
    )

if __name__ == "__main__":
    main()