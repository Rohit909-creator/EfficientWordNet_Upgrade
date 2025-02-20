import os
from Detection import ONNXtoTorchModel
import numpy as np
import librosa
# from tabulate import tabulate
from colorama import init, Fore, Style

init()

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


class WakeWordTester:
    def __init__(self, model_path, threshold=0.59):
        self.model = ONNXtoTorchModel(model_path)
        self.threshold = threshold
        
    def process_audio(self, file_path, expected_length=24000):
        """Process audio file and return embeddings"""
        print(f"Processing: {os.path.basename(file_path)}")
        audio, sr = librosa.load(file_path, sr=16000)
        if len(audio) < expected_length:
            audio = np.pad(audio, (0, expected_length - len(audio)), mode='constant')
        audio = audio[:expected_length]
        return self.model(audio), audio, sr

    def run_comprehensive_test(self, positive_files, negative_files, test_files):
        """Run comprehensive testing with detailed metrics"""
        print(f"\n{Fore.CYAN}=== Wake Word Detection System Test ===\n{Style.RESET_ALL}")
        
        # Process positive examples
        print(f"{Fore.GREEN}Processing positive examples...{Style.RESET_ALL}")
        positive_embeddings = []
        for file in positive_files:
            emb, audio, sr = self.process_audio(file)
            positive_embeddings.append(emb)
        
        # Process negative examples
        print(f"\n{Fore.RED}Processing negative examples...{Style.RESET_ALL}")
        negative_embeddings = []
        for file in negative_files:
            emb, audio, sr = self.process_audio(file)
            negative_embeddings.append(emb)
        
        # Test results
        results = []
        print(f"\n{Fore.YELLOW}Running tests...{Style.RESET_ALL}")
        
        for test_file in test_files:
            # print(f"\nTesting: {os.path.basename(test_file)}")
            emb, audio, sr = self.process_audio(test_file)
            
            # Test against each positive sample
            best_confidence = 0
            best_metrics = None
            
            for pos_emb in positive_embeddings:
                cosine_sim, gaussian_sim, angular_sim, combined_sim = self.model.enhanced_similarity(emb, pos_emb)
                confidence = (cosine_sim + angular_sim) / 2  # Average of cosine and angular
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_metrics = [cosine_sim, gaussian_sim, angular_sim, combined_sim]
            
            # Determine if wake word detected
            is_wake_word = best_confidence > self.threshold
            
            # Add result
            results.append([
                os.path.basename(test_file),
                f"{best_confidence:.4f}",
                "✓" if is_wake_word else "✗",
                f"{best_metrics[0]:.4f}",
                f"{best_metrics[1]:.4f}",
                f"{best_metrics[2]:.4f}",
                f"{best_metrics[3]:.4f}",
                f"{self.estimate_noise_level(audio):.4f}"
            ])
        
        # Print results table
        headers = ["File", "Confidence", "Detection", "Cosine", "Gaussian", "Angular", "Combined", "Noise Level"]
        print(f"\n{Fore.CYAN}=== Test Results ==={Style.RESET_ALL}")
        print(tabulate(headers, results))
        # print(headers)
        # print(results)
        
        # Print statistics
        detections = sum(1 for r in results if r[2] == "✓")
        print(f"\n{Fore.CYAN}=== Statistics ==={Style.RESET_ALL}")
        print(f"Total tests: {len(results)}")
        print(f"Detections: {detections}")
        # print(f"Detection rate: {detections/len(results)*100:.2f}%")
        
        return results

    def estimate_noise_level(self, audio):
        """Estimate noise level in audio signal"""
        signal_power = np.mean(audio ** 2)
        peak_power = np.max(audio ** 2)
        
        if peak_power > 0:
            snr = 10 * np.log10(peak_power / signal_power)
            noise_level = 1 / (1 + np.exp(0.1 * (snr - 10)))
        else:
            noise_level = 1.0
        return noise_level

if __name__ == "__main__":
    # Define paths
    base_dir = "./"
    
    # Model path
    model_path = os.path.join(base_dir, "resnet_50_arc", "slim_93%_accuracy_72.7390%.onnx")
    
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
    
    # Additional test with noisy samples
    # test_files.extend([
    #     os.path.join(base_dir, "tts_samples", "positive", "normal_voice0_noisy.wav"),
    #     os.path.join(base_dir, "tts_samples", "positive", "soft_voice0_noisy.wav")
    # ])
    
    # Run tests
    tester = WakeWordTester(model_path)
    results = tester.run_comprehensive_test(positive_files, negative_files, test_files)