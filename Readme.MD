# EfficientWordNet - Enhanced Similarity Matching

## Overview

### Disclaimer: First Test the wakeword with the default threshold and then adjust it, according to ur needs fr optimal performance



EfficientWordNet, originally developed by AntBrain, has been upgraded with a more advanced similarity measure to improve wakeword detection accuracy. The key enhancement focuses on noise-robust similarity matching, enabling smooth detection in real-world environments without requiring extensive retraining on noise-augmented datasets.

## Key Enhancements

### Improved Similarity Matching

- **Adaptive Similarity Metrics**: Combines multiple similarity metrics to improve detection accuracy.
  - Cosine similarity with a positive centroid
  - Adaptive Gaussian kernel similarity
  - Negative sample penalties
  - Standard deviation-based outlier detection
- **Learned Decision Boundaries**: Utilizes negative samples to refine decision boundaries.
- **Weighted Similarity Metrics**: Dynamically adjusts the importance of different similarity measures based on conditions.

### Noise Robustness

- **Adaptive Gaussian Kernel with Noise-Dependent Sigma**: Models the sensitivity parameter (sigma) as a function of noise levels.
- **Noise Level Estimation**: Extracts noise characteristics from audio signals to adjust similarity weights.
- **Dynamic Weight Adjustment**: In noisy environments, higher emphasis is placed on Gaussian kernel similarity to enhance detection robustness.

## Implementation Details

- **Testings.py**: Contains the main implementation of enhanced similarity matching with noise-based threshold adjustments.
- **Detection.py**: Implements the base model and alternative similarity operations for experimentation and research.

## Future Improvements

- Exploring neural network-based modeling of similarity adjustments for smoother performance.
- Further optimization to balance computational efficiency with detection accuracy.
- Expansion to support more adaptive similarity operations based on real-time environmental conditions.

## How to Use

1. Clone the repository and install the required dependencies.
2. run `pip install -r requirments.txt` 
3. its good to install `ffmpeg` 
4. Use `recording_audio.py` for recording reference embeddings for ur wakeword.
5. Run `main.py` to test the improved similarity matching algorithm in realtime.
6. Use `TabulateTest.py` for baseline model comparison and alternative similarity operations.



This upgrade aims to improve wakeword detection performance in noisy conditions while maintaining efficiency without requiring extensive retraining. Contributions and feedback are welcome!

