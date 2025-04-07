import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
import requests
import json
import os
import time


def enhance_similarity_scores(emb1, emb2, method='enhanced_cosine', cos_sim=None):
    """
    Calculate similarity between embeddings using various methods.
    
    Parameters:
    emb1, emb2: numpy arrays of shape (n_samples, embedding_dim)
    method: str, similarity method to use
    
    Returns:
    float: similarity score between 0 and 1
    """
    
    def enhanced_cosine(e1, e2):
        # L2 normalize embeddings first
        e1_norm = normalize(e1.reshape(1, -1))
        e2_norm = normalize(e2.reshape(1, -1))
        
        # Calculate cosine similarity
        cos_sim = np.matmul(e2_norm, e1_norm.T)
        
        # Apply non-linear transformation to spread out scores
        scaled_sim = np.tanh(2 * cos_sim) * 0.5 + 0.5
        return scaled_sim.max()
    
    def gaussian_kernel(e1, e2, sigma=1.0):
        # Calculate Euclidean distance
        dist = cdist(e1.reshape(1, -1), e2.reshape(1, -1), metric='euclidean')
        # Apply Gaussian kernel
        similarity = np.exp(-dist ** 2 / (2 * sigma ** 2))
        return similarity.max()
    
    def angular_similarity(e1, e2):
        # Convert cosine similarity to angular similarity
        e1_norm = normalize(e1.reshape(1, -1))
        e2_norm = normalize(e2.reshape(1, -1))
        cos_sim = np.matmul(e2_norm, e1_norm.T)
        # Convert to angle (in radians) and normalize to [0,1]
        angular_sim = 1 - np.arccos(np.clip(cos_sim, -1, 1)) / np.pi
        return angular_sim.max()
    
    def combined_similarity(e1, e2, cos_sim):
        # Combine multiple similarity measures
        # cos_sim = enhanced_cosine(e1, e2)
        gauss_sim = gaussian_kernel(e1, e2)
        ang_sim = angular_similarity(e1, e2)
        
        # Weight the different similarities
        # You can adjust these weights based on performance
        weights = [0.4, 0.3, 0.3]
        combined = (weights[0] * cos_sim + 
                   weights[1] * gauss_sim + 
                   weights[2] * ang_sim)
        return combined

    # Dictionary of available methods
    methods = {
        'enhanced_cosine': enhanced_cosine,
        'gaussian': gaussian_kernel,
        'angular': angular_similarity,
        'combined': combined_similarity
    }
    
    if method not in methods:
        raise ValueError(f"Method {method} not supported. Choose from {list(methods.keys())}")
    if method == 'combined':
        return methods[method](emb1, emb2, cos_sim)    
    return methods[method](emb1, emb2)

def preprocess_embeddings(emb):
    """
    Preprocess embeddings to enhance similarity detection
    """
    # Ensure we're working with numpy array
    emb = emb.numpy() if hasattr(emb, 'numpy') else np.array(emb)
    
    # Apply L2 normalization
    emb_normalized = normalize(emb)
    
    # Optional: Remove low-variance dimensions
    # variance = np.var(emb_normalized, axis=0)
    # mask = variance > np.percentile(variance, 10)
    # emb_filtered = emb_normalized[:, mask]
    
    return emb_normalized

    # return emb


def generate_wakewords(wakeword:str):
    # API request for TTS
    
    voice_ids = {"en-US-zion":{"style": "Promo", "multiNativeLocale": "en-US"}, 
                 "en-US-natalie":{"style": "Promo", "multiNativeLocale": "en-US"}, 
                 "en-IN-aarav":{"style": "Conversational", "multiNativeLocale": "en-IN"},
                 "en-IN-alia":{"style": "Promo", "multiNativeLocale": "en-IN"},
                 "en-UK-theo":{"style": "Narration", "multiNativeLocale": "en-UK"},
                 "en-UK-ruby":{"style": "Conversational", "multiNativeLocale": "en-UK"},
                 "en-AU-kylie":{"style": "Conversational", "multiNativeLocale": "en-AU"},
                 "en-AU-jimm":{"style": "Conversational", "multiNativeLocale": "en-AU"}
                #  "zh-CN-tao":{"style": "Conversational", "multiNativeLocale": "zh-CN"},
                #  "zh-CN-jiao":{"style": "Conversational", "multiNativeLocale": "zh-CN"}
                 }
    
    url = "https://api.murf.ai/v1/speech/generate"
    
    
    for key in list(voice_ids.keys()):
    
        style = voice_ids[key]['style']
        multiNativeLocale = voice_ids[key]['multiNativeLocale']
    
        payload = json.dumps({
        "voiceId": key,
        "style": style,
        "text": wakeword,
        "rate": 0,
        "pitch": 0,
        "sampleRate": 48000,
        "format": "MP3",
        "channelType": "MONO",
        "pronunciationDictionary": {},
        "encodeAsBase64": False,
        "variation": 1,
        "audioDuration": 0,
        "modelVersion": "GEN2",
        "multiNativeLocale": multiNativeLocale
        })
        headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'api-key': 'ap2_39e885d1-b227-45bc-b193-c37797a4045c'
        }

        # Make the TTS API request
        response = requests.request("POST", url, headers=headers, data=payload)
        response_data = response.json()

        # Print the full response for debugging
        # print(response.text)
        # print(response_data.keys())

        # Check if the API request was successful and contains the audio file URL
        if response.status_code == 200 and 'audioFile' in response_data:
            # Get the audio file URL
            audio_url = response_data['audioFile']
            
            # Define the output filename (you can customize this)
            output_filename = f"{wakeword}_{key}.mp3"
            
            # Download the audio file
            audio_response = requests.get(audio_url)
            
            # Check if the download was successful
            if audio_response.status_code == 200:
                # Save the audio file
                with open(output_filename, 'wb') as f:
                    f.write(audio_response.content)
                print(f"Audio file downloaded successfully as '{output_filename}'")
            else:
                print(f"Failed to download audio file. Status code: {audio_response.status_code}")
        else:
            print("Failed to generate audio or audio_file URL not found in response")
            print(f"Status code: {response.status_code}")

        time.sleep(0.5)

if __name__ == "__main__":
    
    words = ["Tony", "Alexa", "Sam", "Sneha"]
    for word in words:
        generate_wakewords(word)
        time.sleep(2)


# Results:

# Audio processing:  (24000,) 16000
# Embeddings from audio shape: torch.Size([1, 2048])
# Audio processing:  (24000,) 16000
# Embeddings from audio2 shape: torch.Size([1, 2048])
# Trial with audio files audio2.wav and audio2_twin.wav
# Old Cosine Similarity score: 0.7658122
# Enhanced Similarity Cosine: 0.8792505
# Enhanced Similarity Gaussian: 0.15358484813977985
# Enhanced Similarity Angular: 0.46025136
# Enhanced Similarity Combined: 0.6084176441617999

# Audio processing:  (24000,) 16000
# Embeddings from audio shape: torch.Size([1, 2048])
# Audio processing:  (24000,) 16000
# Embeddings from audio2 shape: torch.Size([1, 2048])
# Trial with audio files audio.wav and audio2_twin.wav
# Old Cosine Similarity score: 0.58687246
# Enhanced Similarity Cosine: 0.5
# Enhanced Similarity Gaussian: 0.03669846499042128
# Enhanced Similarity Angular: 0.30867672
# Enhanced Similarity Combined: 0.3499427133680908

# Audio processing:  (24000,) 16000
# Embeddings from audio shape: torch.Size([1, 2048])
# Audio processing:  (24000,) 16000
# Embeddings from audio2 shape: torch.Size([1, 2048])
# Trial with audio files audio.wav and audio_twin.wav
# Old Cosine Similarity score: 0.6059735
# Enhanced Similarity Cosine: 0.5
# Enhanced Similarity Gaussian: 0.042757439528585475
# Enhanced Similarity Angular: 0.32260182
# Enhanced Similarity Combined: 0.35533204136583824

# Audio processing:  (24000,) 16000
# Embeddings from audio shape: torch.Size([1, 2048])
# Audio processing:  (24000,) 16000
# Embeddings from audio2 shape: torch.Size([1, 2048])
# Trial with audio files audio.wav and audio2_twin.wav
# Old Cosine Similarity score: 0.58687246
# Enhanced Similarity Cosine: 0.5
# Enhanced Similarity Gaussian: 0.03669846499042128
# Enhanced Similarity Angular: 0.30867672
# Enhanced Similarity Combined: 0.3499427133680908
