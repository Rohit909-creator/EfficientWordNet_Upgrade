import os
import nltk
import subprocess
import re
import glob
import random
from nltk.corpus import cmudict
from jellyfish import jaro_winkler_similarity, levenshtein_distance
import numpy as np

# Download CMU Pronouncing Dictionary if not already present
nltk.download('cmudict', quiet=True)
cmu_dict = cmudict.dict()

def get_cmu_phonemes(word):
    """Fetch phonetic transcription from CMU Pronouncing Dictionary."""
    word = word.lower()
    if word in cmu_dict:
        return cmu_dict[word][0]  # CMU dictionary may have multiple pronunciations
    return None

def get_espeak_phonemes(word):
    """Fetch phonetic transcription using espeak-ng."""
    try:
        result = subprocess.run(["espeak-ng", "--ipa", word], capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        print(f"Error with espeak-ng: {e}")
        return None

def calculate_phonetic_similarity(word1, word2):
    """Calculate phonetic similarity between two words using multiple metrics."""
    # Convert to lowercase
    word1, word2 = word1.lower(), word2.lower()
    
    # Direct string similarity
    string_similarity = jaro_winkler_similarity(word1, word2)
    
    # Levenshtein distance (normalized)
    max_len = max(len(word1), len(word2))
    levenshtein_sim = 1 - (levenshtein_distance(word1, word2) / max_len if max_len > 0 else 0)
    
    # Phonetic similarity using CMU dict
    phonemes1 = get_cmu_phonemes(word1)
    phonemes2 = get_cmu_phonemes(word2)
    
    phonetic_sim = 0
    if phonemes1 and phonemes2:
        # Calculate similarity based on phonemes
        # Count matching phonemes in sequence
        matches = 0
        total = max(len(phonemes1), len(phonemes2))
        
        for i in range(min(len(phonemes1), len(phonemes2))):
            # Count as partial match if the consonant/vowel part matches
            if phonemes1[i][0] == phonemes2[i][0]:
                matches += 0.5
                # If the entire phoneme matches
                if phonemes1[i] == phonemes2[i]:
                    matches += 0.5
        
        phonetic_sim = matches / total if total > 0 else 0
    
    # Combine metrics (weighted average)
    combined_similarity = (0.4 * string_similarity + 
                          0.3 * levenshtein_sim + 
                          0.3 * phonetic_sim)
    
    return combined_similarity

def find_wake_word_files(base_dir, pattern=None):
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

def get_similarity_groups(target_word, wake_words, num_similar=2, num_dissimilar=2, num_neutral=2):
    """Group wake words based on their similarity to the target word."""
    similarities = []
    
    for word in wake_words:
        if word.lower() == target_word.lower():
            continue  # Skip the target word itself
        
        similarity = calculate_phonetic_similarity(target_word, word)
        similarities.append((word, similarity))
    
    # Sort by similarity score
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Group by similarity level
    high_similarity = [word for word, score in similarities[:num_similar]]
    low_similarity = [word for word, score in similarities[-num_dissimilar:]]
    
    # Pick some from the middle for neutral similarity
    if len(similarities) > num_similar + num_dissimilar:
        middle_idx = len(similarities) // 2
        start_idx = max(num_similar, middle_idx - num_neutral // 2)
        neutral_similarity = [word for word, score in 
                             similarities[start_idx:start_idx + num_neutral]]
    else:
        neutral_similarity = []
    
    return {
        "high_similarity": high_similarity,
        "neutral_similarity": neutral_similarity,
        "low_similarity": low_similarity
    }

def get_negative_files(base_dir, target_word, similarity_groups, files_per_group=2):
    """Create a list of negative files based on similarity groups."""
    negative_files = []
    all_wake_words, all_files = find_wake_word_files(base_dir)
    
    # Group files by wake word
    word_to_files = {}
    for file_path in all_files:
        filename = os.path.basename(file_path)
        wake_word = filename.split('_')[0]
        
        if wake_word not in word_to_files:
            word_to_files[wake_word] = []
        
        word_to_files[wake_word].append(file_path)
    
    # Add files from each similarity group
    for group_name, words in similarity_groups.items():
        for word in words:
            if word in word_to_files:
                # Get a random sample of files for this word
                word_files = word_to_files[word]
                sample_size = min(files_per_group, len(word_files))
                selected_files = random.sample(word_files, sample_size)
                negative_files.extend(selected_files)
    
    return negative_files

def get_positive_files(base_dir, target_word, variations=None):
    """Get positive files for the target wake word."""
    positive_files = []
    all_wake_words, all_files = find_wake_word_files(base_dir)
    
    # Filter files for the target word
    for file_path in all_files:
        filename = os.path.basename(file_path)
        if filename.startswith(f"{target_word}_"):
            # If variations are specified, check if this file matches
            if variations:
                for variation in variations:
                    if f"_{variation}_" in filename:
                        positive_files.append(file_path)
                        break
            else:
                positive_files.append(file_path)
    
    return positive_files

def search_wake_words(base_dir, target_word, high_similar=2, neutral=2, low_similar=2, files_per_group=2):
    """Main function to search for wake words based on phonetic similarity."""
    # Find all available wake words
    all_wake_words, all_files = find_wake_word_files(base_dir)
    
    print(f"\n🔍 Searching for wake words similar to '{target_word}'")
    print(f"Found {len(all_wake_words)} unique wake words in the dataset")
    
    # Group wake words by similarity
    similarity_groups = get_similarity_groups(
        target_word, all_wake_words, 
        num_similar=high_similar,
        num_neutral=neutral,
        num_dissimilar=low_similar
    )
    
    # Print similarity groups
    print("\n📊 Similarity Analysis:")
    for group, words in similarity_groups.items():
        print(f"\n{group.replace('_', ' ').title()}:")
        for word in words:
            similarity = calculate_phonetic_similarity(target_word, word)
            print(f"  • {word} (similarity score: {similarity:.2f})")
    
    # Get the positive and negative files
    positive_files = get_positive_files(base_dir, target_word)
    negative_files = get_negative_files(base_dir, target_word, similarity_groups, files_per_group)
    
    result = {
        "target_word": target_word,
        "similarity_groups": similarity_groups,
        "positive_files": positive_files,
        "negative_files": negative_files
    }
    
    return result

# Example usage
if __name__ == "__main__":
    base_dir = "."  # Replace with your actual base directory
    target_word = "Ava"
    
    result = search_wake_words(base_dir, target_word)
    
    print("\n✅ Positive Files:")
    for file in result["positive_files"][:5]:  # Show first 5 for brevity
        print(f"  • {os.path.basename(file)}")
    
    print("\n❌ Negative Files (selected based on similarity):")
    for file in result["negative_files"][:10]:  # Show first 10 for brevity
        print(f"  • {os.path.basename(file)}")