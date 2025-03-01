import os
import glob
import WakewordSearch as wws

def test_multiple_wake_words(base_dir, target_words, high_similar=3, neutral=2, low_similar=2, files_per_group=3):
    """
    Test multiple wake words at once using the WakewordSearch algorithm.
    
    Args:
        base_dir: Base directory with wake word data
        target_words: List of wake words to test
        high_similar: Number of high similarity words to include
        neutral: Number of neutral similarity words to include
        low_similar: Number of low similarity words to include
        files_per_group: Number of audio files to select per word
    """
    all_wake_words, all_files = wws.find_wake_word_files(base_dir)
    
    print(f"Found {len(all_wake_words)} unique wake words in the dataset:")
    print(", ".join(all_wake_words))
    
    # Test each target wake word
    for target_word in target_words:
        print(f"\n{'='*60}")
        print(f"Analyzing wake word: '{target_word}'")
        print(f"{'='*60}")
        
        # Skip if not found in dataset
        if target_word not in all_wake_words:
            print(f"Warning: '{target_word}' not found in dataset. Skipping.")
            continue
        
        # Get similarity groups using WakeWordSearch
        search_result = wws.search_wake_words(
            base_dir, 
            target_word, 
            high_similar=high_similar, 
            neutral=neutral, 
            low_similar=low_similar, 
            files_per_group=files_per_group
        )
        
        # Print positive files
        print("\n✅ Positive Files:")
        for file in search_result["positive_files"][:5]:  # Show first 5 for brevity
            print(f"  • {os.path.basename(file)}")
        
        # Print negative files
        print("\n❌ Negative Files (selected based on similarity):")
        for file in search_result["negative_files"]:
            filename = os.path.basename(file)
            word = filename.split('_')[0]
            
            # Determine which group this word belongs to
            group = "unknown"
            for g_name, words in search_result["similarity_groups"].items():
                if word in words:
                    group = g_name
                    break
            
            print(f"  • {filename} ({group})")

def main():
    """Main function to run the multi-wake word tester."""
    try:
        # Configure test parameters
        base_dir = "."  # Replace with the actual base directory of your dataset
        
        # List of wake words to test
        # Edit this list to include the wake words you want to test
        target_words = ["Alexa", "Siri", "Ava"]
        
        # Run the tests
        test_multiple_wake_words(base_dir, target_words)
        
    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()