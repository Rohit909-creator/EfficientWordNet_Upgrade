import sounddevice as sd
import soundfile as sf
import os
import time
from pathlib import Path
import numpy as np

# Audio recording specifications
SAMPLE_RATE = 16000  # 16kHz
CHANNELS = 1        # Mono
TARGET_DURATION = 1.5  # Final duration in seconds
COUNTDOWN_DURATION = 3.0  # 3-second countdown
BUFFER_DURATION = 1.0  # Extra time after "Speak now!" before trimming
FORMAT = 'FLOAT32'    # 32-bit float

def record_audio(filename, wakeword, style_instruction):
    """Record audio and trim to capture wake word after countdown"""
    print(f"\nRecording '{wakeword}' {style_instruction}")
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

def main():
    # Create output directory structure
    base_dir = Path("wake_word_data/recordings")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Get wake word from user
    print("Wake Word Recording Tool")
    while True:
        wakeword = input("Enter your desired wake word: ").strip()
        if wakeword:
            break
        print("Wake word cannot be empty!")
    
    # Recording styles
    recording_styles = [
        ("normal", "normally or casually"),
        ("quick", "quickly"),
        ("shouted", "with a shout"),
        ("whispered", "in a whisper")
    ]
    
    print("\nRecording Setup:")
    print(f"Wake word: {wakeword}")
    print(f"Each recording will be {TARGET_DURATION} seconds long")
    print("A 3-second countdown will help you time your speech")
    print("Starting in 1 second...")
    time.sleep(1)
    
    # Record three samples for each style
    for style_key, style_desc in recording_styles:
        style_dir = base_dir / style_key
        style_dir.mkdir(exist_ok=True)
        
        print(f"\nStyle: {style_key.capitalize()}")
        for i in range(1, 4):
            filename = style_dir / f"{wakeword}_{style_key}_{i}.wav"
            record_audio(filename, wakeword, style_desc)
            time.sleep(0.5)  # Small pause between recordings
    
    print("\nRecording Complete")
    print("Files saved in 'wake_word_data/recordings' with:")
    print(f"- Sample rate: {SAMPLE_RATE}Hz")
    print(f"- Channels: {CHANNELS} (Mono)")
    print(f"- Bit depth: 32-bit float")
    print(f"- Duration: {TARGET_DURATION} seconds")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")