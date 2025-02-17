import torch
import torch.nn as nn
import onnxruntime
import numpy as np
from python_speech_features import logfbank
import librosa
from UTILS import enhance_similarity_scores, preprocess_embeddings

class ONNXtoTorchModel(nn.Module):
    def __init__(self, onnx_path):
        super().__init__()
        # Load ONNX model
        self.session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.window_length = 1.5
        self.window_frames = int(self.window_length * 16000)
        
    def compute_logfbank_features(self, inpAudio):
        """
        Compute log Mel-filterbank features
        """
        return logfbank(
            inpAudio,
            samplerate=16000,
            winlen=0.025,
            winstep=0.01,
            nfilt=64,
            nfft=512,
            preemph=0.0
        )
    
    def get_embeddings(self, audio):
        """
        Convert audio to embeddings
        Args:
            audio: numpy array of shape (window_frames,) - 1.5 seconds of audio at 16kHz
        Returns:
            embeddings: torch tensor of embeddings
        """
        assert audio.shape == (self.window_frames,), f"Expected audio shape {self.window_frames}, got {audio.shape}"
        
        # Compute log mel features
        features = self.compute_logfbank_features(audio)
        
        # Add batch and channel dimensions
        features = np.expand_dims(features, axis=(0,1))
        features = np.float32(features)
        
        # Get embeddings from ONNX model
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: features}
        )[0]
        
        return torch.from_numpy(outputs)
    
    def forward(self, x):
        """
        Forward pass - handles both audio and pre-computed mel spectrograms
        Args:
            x: Either audio waveform or mel spectrogram
        Returns:
            embeddings: torch tensor of embeddings
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            
        if len(x.shape) == 1:  # Audio waveform
            return self.get_embeddings(x)
        else:  # Mel spectrogram
            # Add batch and channel dimensions if needed
            if len(x.shape) == 2:
                x = np.expand_dims(x, axis=(0,1))
            x = np.float32(x)
            outputs = self.session.run([self.output_name], {self.input_name: x})[0]
            return torch.from_numpy(outputs)
        
    def compute_similarity(self, emb1, emb2):
        """
        Compute similarity between two embeddings using cosine similarity
        """
        if isinstance(emb1, torch.Tensor):
            emb1 = emb1.detach().cpu().numpy()
        if isinstance(emb2, torch.Tensor):
            emb2 = emb2.detach().cpu().numpy()
            
        cosine_similarity = np.matmul(emb2, emb1.T)
        confidence_score = (cosine_similarity + 1) / 2
        return confidence_score.max()

    def enhanced_similarity(self, emb1, emb2, test=False):
        
        embs1_processed = preprocess_embeddings(emb1)
        embs2_processed = preprocess_embeddings(emb2)
        if test:
            print("Enhanced Similarity Cosine:", enhance_similarity_scores(embs1_processed, embs2_processed))
            print("Enhanced Similarity Gaussian:", enhance_similarity_scores(embs1_processed, embs2_processed, method="gaussian"))
            print("Enhanced Similarity Angular:", enhance_similarity_scores(embs1_processed, embs2_processed, method="angular"))
            print("Enhanced Similarity Combined:", enhance_similarity_scores(embs1_processed, embs2_processed, method="combined"))

        if not test:
            cosine_sim = enhance_similarity_scores(embs1_processed, embs2_processed)
            gausian_sim = enhance_similarity_scores(embs1_processed, embs2_processed, method="gaussian")
            angular_sim = enhance_similarity_scores(embs1_processed, embs2_processed, method="angular")
            combined_sim = enhance_similarity_scores(embs1_processed, embs2_processed, method="combined")
        
        
        return cosine_sim, gausian_sim, angular_sim, combined_sim


def make_reference(model, name, audio):
    
    import json
    embeddings = model(audio)
    
    d = {"name": name, "embeddings": embeddings.tolist()}
    
    with open("path_to_reference.json", 'w') as f:
        s = json.dumps(d)
        f.write(s)
    

# Example usage:
if __name__ == "__main__":
    model = ONNXtoTorchModel("./resnet_50_arc/slim_93%_accuracy_72.7390%.onnx")
    
    # file_path = "audio2.wav"
    # # Load and resample to 16 kHz
    # audio, sr = librosa.load(file_path, sr=16000)
    # # Ensure audio is exactly 24000 samples long
    # expected_length = 24000
    # if len(audio) < expected_length:
    #     pad_length = expected_length - len(audio)
    #     audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros

    # print("Audio processing: ", audio.shape, sr)
    
    # make_reference(model, "bed", audio)
    
    # # Test with audio
    # audio = np.random.randn(24000)  # 1.5 seconds at 16kHz
    file_path = "audio2.wav"
    # Load and resample to 16 kHz
    audio, sr = librosa.load(file_path, sr=16000)
    # Ensure audio is exactly 24000 samples long
    expected_length = 24000
    if len(audio) < expected_length:
        pad_length = expected_length - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')  # Pad with zeros

    print("Audio processing: ", audio.shape, sr)
    embeddings_audio = model(audio)
    print("Embeddings from audio shape:", embeddings_audio.shape)
    
    # audio = np.random.randn(24000)  # 1.5 seconds at 16kHz
    file_path2 = "audio_twin.wav"
    # Load and resample to 16 kHz
    audio2, sr = librosa.load(file_path2, sr=16000)
    # Ensure audio is exactly 24000 samples long
    expected_length = 24000
    if len(audio2) < expected_length:
        pad_length = expected_length - len(audio2)
        audio2 = np.pad(audio2, (0, pad_length), mode='constant')  # Pad with zeros

    print("Audio processing: ", audio2.shape, sr)
    embeddings_audio2 = model(audio2)
    print("Embeddings from audio2 shape:", embeddings_audio2.shape)
    
    # Test with mel spectrogram
    # mel_spec = np.random.randn(149, 64)  # Example mel spectrogram shape
    # embeddings_mel = model(mel_spec)
    # print("Embeddings from mel spectrogram shape:", embeddings_mel.shape)
    
    # Test similarity computation
    
    print(f"Trial with audio files {file_path} and {file_path2}")
    similarity = model.compute_similarity(embeddings_audio, embeddings_audio2)
    print("Old Cosine Similarity score:", similarity)
    
    model.enhanced_similarity(embeddings_audio, embeddings_audio2)
