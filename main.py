import pyaudio
import numpy as np
import threading
import queue
import time
from typing import Optional, Dict
import json
import os
from Detection import ONNXtoTorchModel  # Import your model class

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
                 threshold: float = 0.75,
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
            
            # Compute similarity with reference embeddings
            # similarity = self.model.compute_similarity(
            #     current_embeddings, 
            #     self.reference_embeddings
            # )
            
            # Enhanced Similarity test
            cosine_sim, gausian_sim, angular_sim, combined_sim = self.model.enhanced_similarity(
                current_embeddings, 
                self.reference_embeddings
            )
            
            
            # Trim buffer to prevent memory growth
            self.audio_buffer = self.audio_buffer[-self.window_samples:]
            
            # Check if similarity exceeds threshold
            if cosine_sim >= self.threshold:
                return {
                    "match": True,
                    "confidence": float(cosine_sim)
                }
                
            if gausian_sim >= self.threshold:
                return {
                    "match": True,
                    "confidence": float(cosine_sim)
                }
                
            if angular_sim >= self.threshold:
                return {
                    "match": True,
                    "confidence": float(cosine_sim)
                }
            
            if combined_sim >= self.threshold:
                return {
                    "match": True,
                    "confidence": float(cosine_sim)
                }
                
            
        return {"match": False, "confidence": 0.0}

def main():
    # Initialize detector with your ONNX model path
    wake_word_detector = HotwordDetector(
        hotword="Hey Assistant",
        reference_file="path_to_reference.json",  # Contains reference embeddings
        model_path=r"C:\Users\Rohit Francis\Desktop\Codes\Pathor Wake Word\EfficientWord-Net\eff_word_net\models\resnet_50_arc\slim_93%_accuracy_72.7390%.onnx",
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