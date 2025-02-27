# record_aditya_samples.py

import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from colorama import Fore, Style, init

# Initialize colorama
init()

# Audio recording specifications (matching recording_audio.py)
SAMPLE_RATE = 16000  # 16kHz
CHANNELS = 1        # Mono
TARGET_DURATION = 1.5  # Final duration in seconds
COUNTDOWN_DURATION = 3.0  # 3-second countdown
BUFFER_DURATION = 1.0  # Extra time after "Speak now!" before trimming
FORMAT = 'FLOAT32'    # 32-bit float

def record_audio(filename, wake_word, style_instruction):
    """Record audio and trim to capture wake word after countdown"""
    print(f"\nRecording '{wake_word}' {style_instruction}")
    print("Press Enter to start the countdown...")
    input()  # Wait for Enter key
    
    # Total recording time: countdown + target duration + buffer
    total_duration = COUNTDOWN_DURATION + TARGET_DURATION + BUFFER_DURATION
    total_samples = int(total_duration * SAMPLE_RATE)
    target_samples = int(TARGET_DURATION * SAMPLE_RATE)
    
    # Start recording
    recording = sd.rec(total_samples, 
                      samplerate=SAMPLE_RATE, 
                      channels=CHANNELS, 
                      dtype='float32')
    print("Recording started in background...")
    
    # Countdown timer
    print("Get ready to speak in:")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("Speak now!")
    
    # Wait for the remaining recording time after "Speak now!"
    time.sleep(TARGET_DURATION + BUFFER_DURATION)
    sd.wait()
    print("Done!")
    
    try:
        # Trim to get the 1.5 seconds starting after the countdown
        start_sample = int(COUNTDOWN_DURATION * SAMPLE_RATE)
        end_sample = start_sample + target_samples
        recording = recording[start_sample:end_sample]
        
        # Basic noise check
        rms = np.sqrt(np.mean(recording**2))
        print(f"Audio RMS (loudness): {rms:.4f}")
        if rms < 0.01:
            print("Warning: Recording might be too quiet or noisy")
        
        # Save the recording as WAV file
        sf.write(str(filename), recording, SAMPLE_RATE, subtype='FLOAT')
        print(f"Saved as: {filename.name}")
    
    except Exception as e:
        print(f"Error during recording: {str(e)}")

def create_aditya_dataset():
    """Create a training and testing dataset specifically for Alexa vs Lexa"""
    # Create output directory structure
    base_dir = Path("aditya_samples")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    positive_dir = base_dir / "positive"
    negative_dir = base_dir / "negative"
    positive_dir.mkdir(exist_ok=True)
    negative_dir.mkdir(exist_ok=True)
    
    # Wake word settings
    wake_word = "Alexa"
    similar_word = "Lexa"
    
    # Recording styles - similar to recording_audio.py
    recording_styles = [
        ("normal", "normally or casually"),
        ("quick", "quickly"),
        ("shouted", "with a shout"),
        ("whispered", "in a whisper")
    ]
    
    print("\n" + "="*50)
    print(f"{Fore.CYAN}Aditya's Wake Word Dataset Creator{Style.RESET_ALL}")
    print(f"Wake word: {wake_word}")
    print(f"Similar word: {similar_word}")
    print(f"Each recording will be {TARGET_DURATION} seconds long")
    print("\nWe'll record multiple variations of both words.")
    
    # Ask user how many samples they want to record
    num_samples = int(input("\nHow many samples of each style do you want to record? (recommended: 3): "))
    
    # Record wake word samples in different styles
    print("\n" + "="*50)
    print(f"{Fore.GREEN}Recording WAKE WORD: '{wake_word}'{Style.RESET_ALL}")
    
    for style_key, style_desc in recording_styles:
        style_dir = positive_dir / style_key
        style_dir.mkdir(exist_ok=True)
        
        print(f"\nStyle: {style_key.capitalize()}")
        for i in range(1, num_samples+1):
            filename = style_dir / f"{wake_word}_{style_key}_{i}.wav"
            record_audio(filename, wake_word, style_desc)
            time.sleep(0.5)  # Small pause between recordings
    
    # Record similar word samples in different styles
    print("\n" + "="*50)
    print(f"{Fore.RED}Recording SIMILAR WORD: '{similar_word}'{Style.RESET_ALL}")
    
    for style_key, style_desc in recording_styles:
        style_dir = negative_dir / style_key
        style_dir.mkdir(exist_ok=True)
        
        print(f"\nStyle: {style_key.capitalize()}")
        for i in range(1, num_samples+1):
            filename = style_dir / f"{similar_word}_{style_key}_{i}.wav"
            record_audio(filename, similar_word, style_desc)
            time.sleep(0.5)  # Small pause between recordings
    
    print("\n" + "="*50)
    print(f"{Fore.CYAN}Dataset creation complete!{Style.RESET_ALL}")
    print(f"Files saved in '{base_dir}' directory with:")
    print(f"- Sample rate: {SAMPLE_RATE}Hz")
    print(f"- Channels: {CHANNELS} (Mono)")
    print(f"- Bit depth: 32-bit float")
    print(f"- Duration: {TARGET_DURATION} seconds")
    
    # Print summary
    total_positive = num_samples * len(recording_styles)
    total_negative = num_samples * len(recording_styles)
    print(f"\nTotal recordings:")
    print(f"- Positive examples ('{wake_word}'): {total_positive}")
    print(f"- Negative examples ('{similar_word}'): {total_negative}")

if __name__ == "__main__":
    try:
        create_aditya_dataset()
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Recording canceled by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}An error occurred: {str(e)}{Style.RESET_ALL}")