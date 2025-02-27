# adaptive_test_alexa.py

import os
import numpy as np
import librosa
import time
import json
from pathlib import Path
from colorama import Fore, Style, init
from adaptive_tester import AdaptiveSimilarityMatcher, SimpleMicStream
from Detection import ONNXtoTorchModel

# Initialize colorama
init()

def alexa_vs_lexa_test(base_dir="./aditya_samples", model_path=None):
    """
    Test the ability to distinguish between 'Alexa' and 'Lexa'
    using the samples recorded by Aditya
    
    Args:
        base_dir: Directory containing the recorded samples
        model_path: Path to ONNX model
    """
    if model_path is None:
        model_path = "./resnet_50_arc/slim_93%_accuracy_72.7390%.onnx"
    
    # Load model
    print(f"\n{Fore.CYAN}=== Alexa vs Lexa Discrimination Test ==={Style.RESET_ALL}")
    print(f"Loading model from {model_path}")
    model = ONNXtoTorchModel(model_path)
    
    # Directory structure
    base_dir = Path(base_dir)
    positive_dir = base_dir / "positive"
    negative_dir = base_dir / "negative"
    
    # Collect all audio files
    positive_files = []
    negative_files = []
    
    # Get positive files (Alexa)
    for style_dir in positive_dir.iterdir():
        if style_dir.is_dir():
            for file in style_dir.glob("*.wav"):
                positive_files.append(str(file))
    
    # Get negative files (Lexa)
    for style_dir in negative_dir.iterdir():
        if style_dir.is_dir():
            for file in style_dir.glob("*.wav"):
                negative_files.append(str(file))
    
    # Print summary of files found
    print(f"Found {len(positive_files)} positive examples (Alexa)")
    print(f"Found {len(negative_files)} negative examples (Lexa)")
    
    if len(positive_files) == 0 or len(negative_files) == 0:
        print(f"{Fore.RED}Error: Not enough audio samples found. Please run record_aditya_samples.py first.{Style.RESET_ALL}")
        return
    
    # Create output directory for results
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Process examples
    print("\n" + "="*50)
    print(f"{Fore.CYAN}Processing example recordings...{Style.RESET_ALL}")
    
    # Load and process positive examples
    positive_embeddings = []
    for file in positive_files:
        audio, sr = librosa.load(file, sr=16000)
        expected_length = 24000
        if len(audio) < expected_length:
            audio = np.pad(audio, (0, expected_length - len(audio)), mode='constant')
        audio = audio[:expected_length]
        print(f"Processing positive example: {os.path.basename(file)}")
        emb = model(audio)
        positive_embeddings.append(emb)
    
    # Load and process negative examples
    negative_embeddings = []
    for file in negative_files:
        audio, sr = librosa.load(file, sr=16000)
        expected_length = 24000
        if len(audio) < expected_length:
            audio = np.pad(audio, (0, expected_length - len(audio)), mode='constant')
        audio = audio[:expected_length]
        print(f"Processing negative example: {os.path.basename(file)}")
        emb = model(audio)
        negative_embeddings.append(emb)
    
    # Initialize matcher
    matcher = AdaptiveSimilarityMatcher(positive_embeddings, negative_embeddings)
    
    # If weights file exists, load it
    weights_file = results_dir / "weights.json"
    if weights_file.exists():
        matcher.load_weights(str(weights_file))
    
    # Initialize history tracking if not already present
    if 'accuracy' not in matcher.history:
        matcher.history['accuracy'] = []
        matcher.history['true_positives'] = []
        matcher.history['false_positives'] = []
    
    # Interactive testing
    print("\n" + "="*50)
    print(f"{Fore.CYAN}Starting interactive testing with feedback!{Style.RESET_ALL}")
    print("In this test, you'll say either 'Alexa' or 'Lexa' and provide feedback on detection.")
    print("This will help the system learn to distinguish between these similar words.")
    
    # Create a microphone stream
    mic_stream = SimpleMicStream()
    mic_stream.start_stream()
    
    # For tracking performance
    test_results = {
        'total': 0,
        'correct': 0,
        'false_positives': 0,
        'false_negatives': 0
    }
    
    try:
        num_tests = int(input("\nHow many test iterations would you like to run? (recommended: 5-10): "))
        
        for test_iteration in range(num_tests):
            print("\n" + "="*50)
            print(f"{Fore.CYAN}Test iteration {test_iteration + 1}/{num_tests}{Style.RESET_ALL}")
            
            # Ask what they'll say
            print("\nFor this test, will you say:")
            print("1. 'Alexa' (should be detected)")
            print("2. 'Lexa' (should NOT be detected)")
            choice = int(input("Enter your choice (1 or 2): "))
            
            should_detect = choice == 1
            word_to_say = "Alexa" if should_detect else "Lexa"
            
            # Collect audio
            print(f"\nYou'll say '{word_to_say}'")
            input("Press Enter when ready to speak...")
            
            audio_buffer = np.array([], dtype=np.float32)
            print(f"{Fore.GREEN}Listening...{Style.RESET_ALL}")
            start_time = time.time()
            
            while time.time() - start_time < 3.0:  # Listen for 3 seconds
                frame = mic_stream.getFrame()
                if frame is not None:
                    audio_buffer = np.append(audio_buffer, frame)
                time.sleep(0.01)  # Small sleep to prevent CPU hogging
            
            # Process the collected audio
            if len(audio_buffer) > 16000:  # At least 1 second of audio
                # Take the middle 1.5 seconds
                mid_point = len(audio_buffer) // 2
                window_samples = 24000  # 1.5 seconds at 16kHz
                start = max(0, mid_point - window_samples // 2)
                audio_window = audio_buffer[start:start + window_samples]
                
                if len(audio_window) < window_samples:
                    audio_window = np.pad(audio_window, (0, window_samples - len(audio_window)), mode='constant')
                
                # Get embeddings
                current_embedding = model(audio_window)
                
                # Determine if wake word detected
                noise_level = matcher.estimate_noise_level(audio_window)
                is_wake, confidence, metrics = matcher.is_wake_word(current_embedding, noise_level)
                
                print(f"\n{Fore.CYAN}=== Detection Results ==={Style.RESET_ALL}")
                print(f"Detection: {'Yes' if is_wake else 'No'}, Confidence: {confidence:.4f}")
                print(f"Expected: {'Yes' if should_detect else 'No'}")
                print(f"Correct: {'✓' if is_wake == should_detect else '✗'}")
                
                # Display detailed metrics
                print(f"\n{Fore.CYAN}=== Similarity Metrics ==={Style.RESET_ALL}")
                for metric, value in metrics.items():
                    print(f"{metric}: {value:.4f}")
                
                # Update test results
                test_results['total'] += 1
                if is_wake == should_detect:
                    test_results['correct'] += 1
                if is_wake and not should_detect:
                    test_results['false_positives'] += 1
                if not is_wake and should_detect:
                    test_results['false_negatives'] += 1
                
                # Automatic weight adjustment based on result
                if is_wake != should_detect:
                    print(f"\n{Fore.YELLOW}Adjusting weights based on result...{Style.RESET_ALL}")
                    matcher.adjust_weights(should_detect, metrics)
                    print("Weights adjusted!")
                    
                    # Save the updated weights
                    matcher.save_weights(str(weights_file))
                else:
                    print(f"\n{Fore.GREEN}Detection was correct! No need to adjust weights.{Style.RESET_ALL}")
                
                # Update history
                current_accuracy = test_results['correct'] / test_results['total']
                matcher.history['accuracy'].append(current_accuracy)
                
                true_positive_rate = 0
                if test_results['total'] - test_results['false_positives'] > 0:
                    true_positive_rate = (test_results['correct'] - test_results['false_positives']) / (test_results['total'] - test_results['false_positives'])
                matcher.history['true_positives'].append(true_positive_rate)
                
                false_positive_rate = 0
                if test_results['false_positives'] + test_results['correct'] - test_results['false_positives'] > 0:
                    false_positive_rate = test_results['false_positives'] / (test_results['false_positives'] + test_results['correct'] - test_results['false_positives'])
                matcher.history['false_positives'].append(false_positive_rate)
                
                # Print current weights
                print(f"\n{Fore.CYAN}=== Current Weights ==={Style.RESET_ALL}")
                for key, value in matcher.weights.items():
                    print(f"{key}: {value:.4f}")
                
                # Print overall performance so far
                print(f"\n{Fore.CYAN}=== Overall Performance ==={Style.RESET_ALL}")
                print(f"Tests completed: {test_results['total']}")
                print(f"Correct detections: {test_results['correct']}")
                print(f"Accuracy: {current_accuracy:.2%}")
            else:
                print(f"{Fore.RED}Not enough audio captured. Please try again.{Style.RESET_ALL}")
                test_iteration -= 1  # Repeat this iteration
        
        # Generate visualization
        print("\n" + "="*50)
        print(f"{Fore.CYAN}Generating visualization of improvements...{Style.RESET_ALL}")
        matcher.visualize_improvement(str(results_dir))
        
        # Final report
        print("\n" + "="*50)
        print(f"{Fore.CYAN}=== Final Performance Report ==={Style.RESET_ALL}")
        print(f"Total tests: {test_results['total']}")
        print(f"Correct detections: {test_results['correct']}")
        print(f"False positives: {test_results['false_positives']}")
        print(f"False negatives: {test_results['false_negatives']}")
        print(f"Overall accuracy: {test_results['correct'] / test_results['total']:.2%}")
        
        # Compare initial and final weights
        print("\n" + "="*50)
        print(f"{Fore.CYAN}=== Weight Changes ==={Style.RESET_ALL}")
        initial_weights = matcher.history['weights'][0]
        final_weights = matcher.weights
        
        print(f"{'Parameter':<15} {'Initial':<10} {'Final':<10} {'Change':<10}")
        print("-" * 45)
        for key in initial_weights:
            change = final_weights[key] - initial_weights[key]
            change_str = f"{change:+.4f}"
            print(f"{key:<15} {initial_weights[key]:<10.4f} {final_weights[key]:<10.4f} {change_str:<10}")
        
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Test interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Error during testing: {str(e)}{Style.RESET_ALL}")
    finally:
        mic_stream.stop_stream()
        
        # Save final state
        matcher.save_weights(str(weights_file))
        
        print(f"\n{Fore.GREEN}Testing completed! Final weights saved to {weights_file}{Style.RESET_ALL}")
        print(f"You can run this test again to continue improving the system.")
        

if __name__ == "__main__":
    try:
        # Check if matplotlib is available for visualization
        try:
            import matplotlib
            matplotlib_available = True
        except ImportError:
            matplotlib_available = False
            print("Warning: matplotlib not found. Visualization will be disabled.")
            print("To enable visualization, install matplotlib: pip install matplotlib")
        
        # Run the alexa vs lexa test
        alexa_vs_lexa_test()
        
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")