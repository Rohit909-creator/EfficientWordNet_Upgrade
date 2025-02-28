import pyaudio
import numpy as np
import threading
import queue
import time
from typing import Optional, Dict
import json
import os
from Detection import ONNXtoTorchModel  # Import your model class
from Detection import EnhancedSimilarityMatcher

class SimpleMicStream:
    """Handles real-time audio capture from microphone"""
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_running = False
        
    def start_stream(self):
        """Start capturing audio from microphone"""
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        self.is_running = True
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def getFrame(self) -> np.ndarray:
        """Get latest audio frame from queue"""
        if not self.is_running:
            return None
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
        
    def stop_stream(self):
        """Stop and clean up audio stream"""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

class HotwordDetector:
    """Wake word detector using similarity-based detection"""
    def __init__(self, 
                 hotword: str,
                 reference_file: str,
                 model_path: str,
                 matcher: EnhancedSimilarityMatcher,
                 threshold: float = 0.60,
                 window_length: float = 1.5):
        self.hotword = hotword
        self.threshold = threshold
        self.window_length = window_length
        self.sample_rate = 16000
        self.window_samples = int(self.window_length * self.sample_rate)
        
        # Load reference embeddings
        self.reference_embeddings = self._load_reference_embeddings(reference_file)
        
        # Initialize model
        self.model = ONNXtoTorchModel(model_path)
        
        # Store the matcher
        self.matcher = matcher
        
        # Buffer for collecting audio frames
        self.audio_buffer = np.array([], dtype=np.float32)
        
    def _load_reference_embeddings(self, reference_file: str) -> np.ndarray:
        """Load reference embeddings from JSON file"""
        with open(reference_file, 'r') as f:
            data = json.load(f)
        return np.array(data['embeddings'])
    
    def scoreFrame(self, frame: np.ndarray) -> Optional[Dict]:
        """Process audio frame and check for wake word"""
        if frame is None:
            return None
            
        # Add frame to buffer
        self.audio_buffer = np.append(self.audio_buffer, frame)
        
        # If we have enough audio, process it
        if len(self.audio_buffer) >= self.window_samples:
            # Take the last window_samples
            audio_window = self.audio_buffer[-self.window_samples:]
            
            # Get embeddings for current audio
            current_embeddings = self.model(audio_window)
            
            # Use the matcher to determine if this is a wake word
            noise_level = self.matcher.estimate_noise_level(audio_window)
            
            is_wake_word, confidence, similarities = self.matcher.is_wake_word(current_embeddings, noise_level)
            
            # Trim buffer to prevent memory growth
            self.audio_buffer = self.audio_buffer[-self.window_samples:]
            
            # Check if confidence exceeds threshold
            if is_wake_word and confidence >= self.threshold:
                return {
                    "match": True,
                    "confidence": float(confidence),
                    "similarities": similarities
                }
                
        return {"match": False, "confidence": 0.0, "similarities": {}}

def main():
    
    import librosa
    from colorama import Fore, Style
    
    base_dir = "./"
    
    model_path = os.path.join(base_dir, "resnet_50_arc", "slim_93%_accuracy_72.7390%.onnx")
    model = ONNXtoTorchModel(model_path)
    
    
    dir_list = os.listdir(os.path.join(base_dir, "wake_word_data", "recordings"))
    
    # positive_files = []
    
    # for folder in dir_list:
    #     audio_files = os.listdir(os.path.join(base_dir, "wake_word_data", "recordings", folder))
    #     for audio_file in audio_files:
    #         positive_files.append(os.path.join(base_dir, "wake_word_data", "recordings", folder, audio_file))
     
    
    positive_files = [
        os.path.join(base_dir, "wake_word_data", "recordings", "normal", "Alexa_normal_1.wav"),
        os.path.join(base_dir, "wake_word_data", "recordings", "quick", "Alexa_quick_1.wav"),
        os.path.join(base_dir, "wake_word_data", "recordings", "shouted", "Alexa_shouted_1.wav"),
        os.path.join(base_dir, "wake_word_data", "recordings", "whispered", "Alexa_whispered_1.wav"),
        os.path.join(base_dir, "wake_word_data", "recordings", "normal", "Alexa_normal_2.wav"),
        os.path.join(base_dir, "wake_word_data", "recordings", "normal", "Alexa_normal_3.wav")
    ]
    
    # negative_files = [
    #     os.path.join(base_dir, "tts_samples", "negative", "partial_voice0.wav"),
    #     os.path.join(base_dir, "tts_samples", "negative", "partial_voice1.wav"),
    #     os.path.join(base_dir, "tts_samples", "negative", "last_part_voice0.wav"),
    #     os.path.join(base_dir, "tts_samples", "negative", "last_part_voice1.wav")
    # ]
    
    
    negative_files = [
        os.path.join(base_dir, "wake_word_data", "recordings", "normal", "Hello_normal_1.wav"),
        os.path.join(base_dir, "wake_word_data", "recordings", "quick", "Hello_quick_1.wav"),
        os.path.join(base_dir, "wake_word_data", "recordings", "normal", "Eliza_normal_2.wav"),
        os.path.join(base_dir, "wake_word_data", "recordings", "shouted", "Eliza_shouted_3.wav"),
        os.path.join(base_dir, "wake_word_data", "recordings", "whispered", "Eliza_whispered_3.wav"),
        os.path.join(base_dir, "wake_word_data", "recordings", "normal", "Jeeva_normal_1.wav"),
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
    
    # Initialize detector with your ONNX model path
    wake_word_detector = HotwordDetector(
        hotword="Alexa",
        reference_file="path_to_reference.json",  # Contains reference embeddings
        model_path="./resnet_50_arc/slim_93%_accuracy_72.7390%.onnx",
        matcher=matcher,
        window_length=1.5,
        threshold=0.5  # Adjust based on your needs
    )
    
    # Start microphone stream
    mic_stream = SimpleMicStream()
    mic_stream.start_stream()
    
    print(f"Listening for wake word '{wake_word_detector.hotword}'...")
    try:
        while True:
            frame = mic_stream.getFrame()
            result = wake_word_detector.scoreFrame(frame)
            # print(frame)
            if result is None:
                continue
                
            if result["match"]:
                print(f"Wake word detected! (confidence: {result['confidence']:.2f})")
                # Add your callback action here
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mic_stream.stop_stream()

if __name__ == "__main__":
    main()