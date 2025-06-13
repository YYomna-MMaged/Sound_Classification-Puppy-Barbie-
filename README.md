# Barbie vs Puppy Voice Classifier

This project implements a machine learning system that classifies audio clips as either "Barbie" or "Puppy" voices using acoustic feature extraction and logistic regression with bagging ensemble.

## Features
- Audio preprocessing pipeline (silence removal, pre-emphasis, normalization)
- Feature extraction (MFCCs, spectral features, zero-crossing rate)
- Logistic Regression classifier with Bagging ensemble
- Model persistence and prediction interface

### Technical Approach

#### Audio Preprocessing
1. **Silence Removal**:  
   Trims leading/trailing silence using dB threshold detection
2. **Pre-emphasis**:  
   Applies high-pass filter (α=0.97) to boost high frequencies
3. **Normalization**:  
   Scales amplitude to consistent range [-1, 1]

#### Feature Extraction
| Feature Type             | Description                                  | Dimensions |
|--------------------------|----------------------------------------------|------------|
| MFCCs                    | Mel-frequency cepstral coefficients          | 13         |
| Delta MFCCs              | First-order temporal differences             | 13         |
| Delta-Delta MFCCs        | Second-order temporal differences            | 13         |
| Zero-Crossing Rate (ZCR) | Rate of sign changes in audio waveform       | 1          |
| Spectral Centroid        | Center of mass of the spectrum               | 1          |
| Spectral Rolloff         | Frequency below which 85% of energy resides  | 1          |

#### Machine Learning Pipeline
1. **Feature Aggregation**:  
   Temporal features summarized using statistical measures (mean/std)
   
2. **Label Encoding**:  
   Converts text labels ("barbie"/"puppy") → numerical values (0/1)

3. **Standard Scaling**:  
   Normalizes features to μ=0, σ=1 using `StandardScaler`

4. **Bagging Ensemble**:
   - 20 base Logistic Regression models
   - Bootstrap sampling creates diverse training subsets
   - Majority voting determines final prediction