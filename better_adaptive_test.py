# enhanced_adaptive_test_alexa.py

import os
import numpy as np
import librosa
import time
import json
from pathlib import Path
from colorama import Fore, Style, init
from adaptive_tester import AdaptiveSimilarityMatcher, SimpleMicStream
from Detection import ONNXtoTorchModel

# Try to import embedding-based search
try:
    from enhanced_embedding_wakeword_search_no_gender import search_wake_words_embeddings
    embedding_search_available = True
except ImportError:
    embedding_search_available = False
    print("Warning: enhanced_embedding_wakeword_search_no_gender.py not found.")
    print("Advanced similarity search will be disabled.")

# Initialize colorama
init()

def enhanced_noise_analysis(audio):
    """More detailed noise analysis"""
    # Basic noise level calculation
    signal_power = np.mean(audio ** 2)
    peak_power = np.max(audio ** 2)
    
    if peak_power > 0:
        snr = 10 * np.log10(peak_power / signal_power)
        noise_level = 1 / (1 + np.exp(0.1 * (snr - 10)))
    else:
        snr = 0
        noise_level = 1.0
        
    # Additional noise metrics
    # Zero crossing rate - higher for noisy signals
    zero_crossing_rate = np.mean(np.abs(np.diff(np.signbit(audio))))
    
    # Spectral flatness - higher for noise, lower for tonal sounds
    spectrum = np.abs(np.fft.rfft(audio)) + 1e-10
    spectral_flatness = np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum)
    
    # Energy distribution - check if energy is concentrated or spread out
    energy_distribution = np.std(spectrum) / np.mean(spectrum)
    
    # Combined noise score with weighted metrics
    combined_noise = (
        0.6 * noise_level + 
        0.2 * zero_crossing_rate + 
        0.1 * spectral_flatness +
        0.1 * (1 - min(1, energy_distribution / 10))  # Lower is noisier
    )
    
    return {
        'overall': np.clip(combined_noise, 0, 1),
        'basic_level': np.clip(noise_level, 0, 1),
        'snr': snr,
        'zero_crossing_rate': zero_crossing_rate,
        'spectral_flatness': spectral_flatness,
        'energy_distribution': energy_distribution
    }

def alexa_vs_lexa_test(base_dir="./aditya_samples", model_path=None, use_similarity_search=True):
    """
    Test the ability to distinguish between 'Alexa' and 'Lexa'
    using the samples recorded by Aditya
    
    Args:
        base_dir: Directory containing the recorded samples
        model_path: Path to ONNX model
        use_similarity_search: Whether to use embedding-based similarity search for challenging samples
    """
    if model_path is None:
        model_path = "./resnet_50_arc/slim_93%_accuracy_72.7390%.onnx"
    
    # Load model
    print(f"\n{Fore.CYAN}=== Enhanced Alexa vs Lexa Discrimination Test ==={Style.RESET_ALL}")
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
    positive_audios = []
    for file in positive_files:
        audio, sr = librosa.load(file, sr=16000)
        expected_length = 24000
        if len(audio) < expected_length:
            audio = np.pad(audio, (0, expected_length - len(audio)), mode='constant')
        audio = audio[:expected_length]
        print(f"Processing positive example: {os.path.basename(file)}")
        emb = model(audio)
        positive_embeddings.append(emb)
        positive_audios.append(audio)
        
        # Analyze noise level for this sample
        noise_metrics = enhanced_noise_analysis(audio)
        print(f"  Noise level: {noise_metrics['overall']:.2f}, SNR: {noise_metrics['snr']:.2f} dB")
    
    # Load and process negative examples
    negative_embeddings = []
    negative_audios = []
    for file in negative_files:
        audio, sr = librosa.load(file, sr=16000)
        expected_length = 24000
        if len(audio) < expected_length:
            audio = np.pad(audio, (0, expected_length - len(audio)), mode='constant')
        audio = audio[:expected_length]
        print(f"Processing negative example: {os.path.basename(file)}")
        emb = model(audio)
        negative_embeddings.append(emb)
        negative_audios.append(audio)
        
        # Analyze noise level for this sample
        noise_metrics = enhanced_noise_analysis(audio)
        print(f"  Noise level: {noise_metrics['overall']:.2f}, SNR: {noise_metrics['snr']:.2f} dB")
    
    # Find challenging samples using embedding similarity if available and requested
    challenging_files = []
    if use_similarity_search and embedding_search_available:
        print(f"\n{Fore.CYAN}Finding challenging samples using embedding similarity...{Style.RESET_ALL}")
        
        # Get all available wake words
        wake_word_dir = "./wake_word_data/recordings"
        
        if os.path.exists(wake_word_dir):
            try:
                # Find challenging similar wake words
                result = search_wake_words_embeddings(
                    ".", 
                    "Alexa", 
                    model_path,
                    num_similar=3,
                    files_per_group=2
                )
                
                if result and 'negative_files' in result:
                    # Add high similarity examples to negative_embeddings
                    high_sim_files = [f for f in result['negative_files'] 
                                    if os.path.basename(f).split('_')[0] in result['similarity_groups']['high_similarity']]
                    
                    for file in high_sim_files[:3]:  # Add up to 3 challenging samples
                        audio, sr = librosa.load(file, sr=16000)
                        expected_length = 24000
                        if len(audio) < expected_length:
                            audio = np.pad(audio, (0, expected_length - len(audio)), mode='constant')
                        audio = audio[:expected_length]
                        print(f"Adding challenging negative: {os.path.basename(file)}")
                        
                        # Analyze noise level for this sample
                        noise_metrics = enhanced_noise_analysis(audio)
                        print(f"  Noise level: {noise_metrics['overall']:.2f}, SNR: {noise_metrics['snr']:.2f} dB")
                        
                        emb = model(audio)
                        negative_embeddings.append(emb)
                        negative_audios.append(audio)
                        challenging_files.append(file)
                    
                    print(f"Added {len(challenging_files)} challenging samples to improve discrimination")
            except Exception as e:
                print(f"{Fore.RED}Error finding challenging samples: {str(e)}{Style.RESET_ALL}")
                print("Continuing without challenging samples...")
    
    # Initialize matcher with noise level history
    noise_levels = [enhanced_noise_analysis(audio)['overall'] for audio in positive_audios + negative_audios]
    
    # Check if noise_levels parameter is accepted by AdaptiveSimilarityMatcher
    try:
        matcher = AdaptiveSimilarityMatcher(positive_embeddings, negative_embeddings, noise_levels)
    except TypeError:
        # Fall back to original constructor if noise_levels not supported
        matcher = AdaptiveSimilarityMatcher(positive_embeddings, negative_embeddings)
    
    # If weights file exists, load it
    weights_file = results_dir / "weights.json"
    if weights_file.exists():
        matcher.load_weights(str(weights_file))
    
    # Initialize history tracking if not already present
    if not hasattr(matcher, 'history') or matcher.history is None:
        matcher.history = {}
    
    if 'accuracy' not in matcher.history:
        matcher.history['accuracy'] = []
        matcher.history['true_positives'] = []
        matcher.history['false_positives'] = []
        matcher.history['noise_vs_accuracy'] = []
    
    # Interactive testing
    print("\n" + "="*50)
    print(f"{Fore.CYAN}Starting enhanced interactive testing with feedback!{Style.RESET_ALL}")
    print("In this test, you'll say either 'Alexa' or 'Lexa' and provide feedback on detection.")
    print("This will help the system learn to distinguish between these similar words.")
    print("The enhanced version includes noise analysis and noise-adapted thresholds.")
    
    # Create a microphone stream
    mic_stream = SimpleMicStream()
    mic_stream.start_stream()
    
    # For tracking performance
    test_results = {
        'total': 0,
        'correct': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'noise_levels': [],
        'detection_success': []  # 1 for correct, 0 for incorrect
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
                
                # Enhanced noise analysis
                noise_metrics = enhanced_noise_analysis(audio_window)
                print(f"\n{Fore.YELLOW}=== Enhanced Noise Analysis ==={Style.RESET_ALL}")
                print(f"Overall noise level: {noise_metrics['overall']:.4f}")
                print(f"Base noise level: {noise_metrics['basic_level']:.4f}")
                print(f"Signal-to-noise ratio: {noise_metrics['snr']:.2f} dB")
                print(f"Zero crossing rate: {noise_metrics['zero_crossing_rate']:.4f}")
                print(f"Spectral flatness: {noise_metrics['spectral_flatness']:.4f}")
                print(f"Energy distribution: {noise_metrics['energy_distribution']:.4f}")
                
                # Get embeddings
                current_embedding = model(audio_window)
                
                # Determine if wake word detected with enhanced noise level
                is_wake, confidence, metrics = matcher.is_wake_word(current_embedding, noise_metrics['overall'])
                
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
                    test_results['detection_success'].append(1)
                else:
                    test_results['detection_success'].append(0)
                    
                if is_wake and not should_detect:
                    test_results['false_positives'] += 1
                if not is_wake and should_detect:
                    test_results['false_negatives'] += 1
                
                # Store noise level for analysis
                test_results['noise_levels'].append(noise_metrics['overall'])
                
                # Automatic weight adjustment based on result
                if is_wake != should_detect:
                    print(f"\n{Fore.YELLOW}Adjusting weights based on result...{Style.RESET_ALL}")
                    
                    # Try to pass noise level to the weight adjustment
                    # If it fails, fall back to original method
                    try:
                        # First try with noise_level parameter
                        matcher.adjust_weights(should_detect, metrics, noise_level=noise_metrics['overall'])
                    except TypeError:
                        # Fall back to original method if noise_level is not supported
                        matcher.adjust_weights(should_detect, metrics)
                    
                    print("Weights adjusted!")
                    
                    # Save the updated weights
                    matcher.save_weights(str(weights_file))
                else:
                    print(f"\n{Fore.GREEN}Detection was correct! No need to adjust weights.{Style.RESET_ALL}")
                
                # Update history
                current_accuracy = test_results['correct'] / test_results['total']
                matcher.history['accuracy'].append(current_accuracy)
                
                # Track true positive and false positive rates
                try:
                    true_positive_rate = 0
                    if test_results['total'] - test_results['false_positives'] > 0:
                        true_positive_rate = (test_results['correct'] - test_results['false_positives']) / (test_results['total'] - test_results['false_positives'])
                    matcher.history['true_positives'].append(true_positive_rate)
                    
                    false_positive_rate = 0
                    if test_results['false_positives'] + test_results['correct'] - test_results['false_positives'] > 0:
                        false_positive_rate = test_results['false_positives'] / (test_results['false_positives'] + (test_results['correct'] - test_results['false_positives']))
                    matcher.history['false_positives'].append(false_positive_rate)
                except:
                    # Skip if calculation fails
                    pass
                
                # Add noise vs accuracy data point
                if 'noise_vs_accuracy' in matcher.history:
                    matcher.history['noise_vs_accuracy'].append((noise_metrics['overall'], 1 if is_wake == should_detect else 0))
                
                # Print current weights
                print(f"\n{Fore.CYAN}=== Current Weights ==={Style.RESET_ALL}")
                for key, value in matcher.weights.items():
                    print(f"{key}: {value:.4f}")
                
                # Print overall performance so far
                print(f"\n{Fore.CYAN}=== Overall Performance ==={Style.RESET_ALL}")
                print(f"Tests completed: {test_results['total']}")
                print(f"Correct detections: {test_results['correct']}")
                print(f"Accuracy: {current_accuracy:.2%}")
                
                # Noise performance analysis
                if len(test_results['noise_levels']) > 2:
                    try:
                        # Check correlation between noise and detection success
                        correlation = np.corrcoef(test_results['noise_levels'], test_results['detection_success'])[0, 1]
                        print(f"Noise-accuracy correlation: {correlation:.2f}")
                        
                        if correlation < -0.3:
                            print(f"{Fore.YELLOW}Warning: Detection accuracy decreases with higher noise{Style.RESET_ALL}")
                        elif correlation > 0.3:
                            print(f"{Fore.GREEN}Note: System seems more accurate in noisy conditions{Style.RESET_ALL}")
                    except:
                        # If correlation calculation fails, just skip it
                        pass
            else:
                print(f"{Fore.RED}Not enough audio captured. Please try again.{Style.RESET_ALL}")
                test_iteration -= 1  # Repeat this iteration
        
        # Generate visualization
        print("\n" + "="*50)
        print(f"{Fore.CYAN}Generating visualization of improvements...{Style.RESET_ALL}")
        
        # Check if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            
            # Check if visualize_improvement method exists
            if hasattr(matcher, 'visualize_improvement'):
                # Save the original visualization
                try:
                    matcher.visualize_improvement(str(results_dir))
                except Exception as e:
                    print(f"{Fore.YELLOW}Error generating standard visualization: {str(e)}{Style.RESET_ALL}")
            
            # Add a new visualization for noise vs accuracy
            if 'noise_vs_accuracy' in matcher.history and len(matcher.history['noise_vs_accuracy']) > 3:
                try:
                    plt.figure(figsize=(10, 6))
                    noise_levels = [point[0] for point in matcher.history['noise_vs_accuracy']]
                    accuracies = [point[1] for point in matcher.history['noise_vs_accuracy']]
                    
                    plt.scatter(noise_levels, accuracies, alpha=0.7)
                    
                    # Add regression line
                    z = np.polyfit(noise_levels, accuracies, 1)
                    p = np.poly1d(z)
                    plt.plot(sorted(noise_levels), p(sorted(noise_levels)), "r--", alpha=0.7)
                    
                    plt.xlabel('Noise Level')
                    plt.ylabel('Detection Success (1=correct, 0=incorrect)')
                    plt.title('Effect of Noise on Detection Accuracy')
                    plt.ylim(-0.1, 1.1)
                    plt.grid(True, alpha=0.3)
                    
                    # Add correlation coefficient
                    correlation = np.corrcoef(noise_levels, accuracies)[0, 1]
                    plt.annotate(f"Correlation: {correlation:.2f}", xy=(0.05, 0.95), xycoords='axes fraction')
                    
                    plt.savefig(os.path.join(results_dir, "noise_vs_accuracy.png"))
                    plt.close()
                    
                    print(f"Noise analysis visualization saved to {os.path.join(results_dir, 'noise_vs_accuracy.png')}")
                except Exception as e:
                    print(f"{Fore.YELLOW}Error generating noise visualization: {str(e)}{Style.RESET_ALL}")
        except ImportError:
            print(f"{Fore.YELLOW}Matplotlib not available. Skipping visualizations.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}General error in visualization: {str(e)}{Style.RESET_ALL}")
        
        # Final report
        print("\n" + "="*50)
        print(f"{Fore.CYAN}=== Final Performance Report ==={Style.RESET_ALL}")
        print(f"Total tests: {test_results['total']}")
        print(f"Correct detections: {test_results['correct']}")
        print(f"False positives: {test_results['false_positives']}")
        print(f"False negatives: {test_results['false_negatives']}")
        print(f"Overall accuracy: {test_results['correct'] / test_results['total']:.2%}")
        
        # Noise analysis summary
        if test_results['noise_levels']:
            avg_noise = np.mean(test_results['noise_levels'])
            print(f"\n{Fore.CYAN}=== Noise Analysis Summary ==={Style.RESET_ALL}")
            print(f"Average noise level: {avg_noise:.4f}")
            print(f"Min noise level: {min(test_results['noise_levels']):.4f}")
            print(f"Max noise level: {max(test_results['noise_levels']):.4f}")
            
            # Split results by noise level
            low_noise_idx = [i for i, n in enumerate(test_results['noise_levels']) if n < avg_noise]
            high_noise_idx = [i for i, n in enumerate(test_results['noise_levels']) if n >= avg_noise]
            
            low_noise_success = [test_results['detection_success'][i] for i in low_noise_idx]
            high_noise_success = [test_results['detection_success'][i] for i in high_noise_idx]
            
            low_noise_accuracy = np.mean(low_noise_success) if low_noise_success else 0
            high_noise_accuracy = np.mean(high_noise_success) if high_noise_success else 0
            
            print(f"Accuracy in low noise conditions: {low_noise_accuracy:.2%}")
            print(f"Accuracy in high noise conditions: {high_noise_accuracy:.2%}")
            
            # Guidance based on analysis
            print(f"\n{Fore.CYAN}=== Recommendations ==={Style.RESET_ALL}")
            if low_noise_accuracy > high_noise_accuracy + 0.1:
                print("- System performs better in quiet environments")
                print("- Consider further adjusting weights to improve high-noise performance")
            elif high_noise_accuracy > low_noise_accuracy + 0.1:
                print("- System performs better in noisy environments")
                print("- Consider adjusting weights to improve performance in quiet settings")
            else:
                print("- System performs consistently across noise conditions")
        
        # Compare initial and final weights
        print("\n" + "="*50)
        print(f"{Fore.CYAN}=== Weight Changes ==={Style.RESET_ALL}")
        if hasattr(matcher, 'history') and 'weights' in matcher.history and matcher.history['weights']:
            initial_weights = matcher.history['weights'][0]
            final_weights = matcher.weights
            
            print(f"{'Parameter':<15} {'Initial':<10} {'Final':<10} {'Change':<10}")
            print("-" * 45)
            for key in initial_weights:
                change = final_weights[key] - initial_weights[key]
                change_str = f"{change:+.4f}"
                print(f"{key:<15} {initial_weights[key]:<10.4f} {final_weights[key]:<10.4f} {change_str:<10}")
        else:
            print("No weight history available.")
        
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Test interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Error during testing: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
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
        import traceback
        traceback.print_exc()