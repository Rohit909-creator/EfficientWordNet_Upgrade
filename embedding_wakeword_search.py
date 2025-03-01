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
    
    return embedding.detach().cpu().numpy()

def calculate_similarity(emb1, emb2):
    """Calculate similarity between two embeddings"""
    # Reshape if needed
    if len(emb1.shape) > 1:
        emb1 = emb1.squeeze()
    if len(emb2.shape) > 1:
        emb2 = emb2.squeeze()
        
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
    
    # Convert to a 0-1 scale for easier interpretation
    scaled_sim = (cosine_sim + 1) / 2
    
    return scaled_sim

def search_wake_words_embeddings(base_dir, target_word, model_path, num_similar=3, files_per_group=2):
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
    print(f"\n{Fore.CYAN}🔍 Searching for wake words similar to '{target_word}' using embeddings{Style.RESET_ALL}")
    
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
    for file in target_files:
        embedding = get_wake_word_embeddings(model, file)
        target_embeddings.append(embedding)
    
    # Average the embeddings to get a representative embedding for the target word
    target_embedding = np.mean(target_embeddings, axis=0)
    
    # Calculate similarity to other wake words
    print(f"{Fore.YELLOW}Calculating similarities to other wake words...{Style.RESET_ALL}")
    similarities = []
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
        for file in sample_files:
            embedding = get_wake_word_embeddings(model, file)
            word_embeddings.append(embedding)
        
        # Average the embeddings
        word_embedding = np.mean(word_embeddings, axis=0)
        
        # Calculate similarity
        similarity = calculate_similarity(target_embedding, word_embedding)
        similarities.append((word, similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Group by similarity level
    high_similarity = similarities[:min(num_similar, len(similarities))]
    
    # Calculate threshold for medium similarity (40% of the range from lowest to highest)
    if len(similarities) > num_similar:
        max_sim = similarities[0][1]
        min_sim = similarities[-1][1]
        threshold = min_sim + 0.4 * (max_sim - min_sim)
        
        # Find medium similarity words
        medium_similarity = [(word, sim) for word, sim in similarities[num_similar:] 
                             if sim >= threshold]
        medium_similarity = medium_similarity[:min(num_similar, len(medium_similarity))]
        
        # Low similarity (everything else, limited to num_similar)
        low_similarity = [(word, sim) for word, sim in similarities 
                          if sim < threshold and (word, sim) not in high_similarity]
        low_similarity = low_similarity[-min(num_similar, len(low_similarity)):]
    else:
        # Not enough words for medium/low groups
        medium_similarity = []
        low_similarity = []
    
    # Print results
    print(f"\n{Fore.CYAN}📊 Similarity Analysis (Embedding-Based):{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}High Similarity:{Style.RESET_ALL}")
    for word, score in high_similarity:
        print(f"  • {word} (similarity score: {score:.2f})")
    
    if medium_similarity:
        print(f"\n{Fore.YELLOW}Neutral Similarity:{Style.RESET_ALL}")
        for word, score in medium_similarity:
            print(f"  • {word} (similarity score: {score:.2f})")
    
    if low_similarity:
        print(f"\n{Fore.RED}Low Similarity:{Style.RESET_ALL}")
        for word, score in low_similarity:
            print(f"  • {word} (similarity score: {score:.2f})")
    
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
        sample_size = min(files_per_group, len(word_files))
        negative_files.extend(word_files[:sample_size])
    
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
    parser = argparse.ArgumentParser(description="Search for similar wake words using embeddings")
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