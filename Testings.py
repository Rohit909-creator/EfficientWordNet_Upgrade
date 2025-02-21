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

