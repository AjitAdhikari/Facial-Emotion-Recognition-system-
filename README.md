# Facial Emotion Recognition (FER) System

## Overview

Facial Emotion Recognition (FER) is a machine learning-based system that detects and classifies human emotions from facial expressions in real time. This project is designed to enhance human-computer interaction (HCI) across various fields such as healthcare, education, security, and accessibility for individuals with speech and hearing impairments.

## Features

- Real-time facial emotion detection using a webcam
- Classification of multiple emotions (e.g., Happy, Sad, Angry, Neutral, etc.)
- Robust model trained with Convolutional Neural Networks (CNNs)
- Support for different lighting conditions and slight occlusions
- Future enhancements: 3D face reconstruction, multi-angle training, lightweight model optimization for edge devices

## Technologies Used

- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow / Keras
- **Computer Vision**: OpenCV
- **GUI Development**: Tkinter / PyQt (Planned)
- **Hardware Compatibility**: Raspberry Pi, NVIDIA Jetson (Planned)

## Installation

### Prerequisites

Ensure you have the following dependencies installed before proceeding:

- Python 3.8+
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib
- dlib (for facial landmark detection)
- Scikit-learn

### Steps to Install

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/FER-System.git
    cd FER-System
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv fer_env
    source fer_env/bin/activate  # On Windows: fer_env\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:
    ```bash
    realtimedetector.py
    ```

## Usage

- Run the script to launch the FER system.
- The webcam will start capturing real-time video.
- Detected facial emotions will be displayed on the screen.
- *(Future)* Users can configure settings, view emotion history, or provide manual corrections.

## Model Training

- The CNN model is trained on a dataset of facial expressions.
- Data augmentation techniques such as flipping, rotation, and histogram equalization are used to improve performance.
- Training scripts and dataset details are available in the `training` folder.

## Performance

- **Validation Accuracy**: ~93%
- **Challenges**: Slight inconsistencies in real-time emotion recognition due to lighting, occlusions, and overlapping expressions.

### Planned Improvements:
- Enhanced pre-processing techniques
- Optimized lightweight model for edge devices
- Improved emotion detection under varied conditions

## Future Enhancements

- Implement a user-friendly GUI for real-time analysis and settings control.
- Integrate head pose estimation and 3D face reconstruction for better accuracy.
- Optimize the model for mobile and embedded devices.
- Expand emotion categories for more detailed recognition.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (feature-branch).
3. Commit your changes.
4. Submit a pull request.


## Contact

For any questions or collaboration opportunities, feel free to reach out via ajit07adhikari@gmail.com.
