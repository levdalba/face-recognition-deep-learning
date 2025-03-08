# Face Recognition Deep Learning

A facial verification system built with TensorFlow and Siamese Networks that can verify a person's identity using webcam input.

## Project Overview

This project implements a facial verification system using Siamese Neural Networks. The system compares an input face image (captured in real-time from a webcam) against a set of verification images to determine if the person is verified/authenticated.

### Key Features

- Real-time facial verification using webcam input
- Custom Siamese Network architecture with L1 distance metric
- Training pipeline with positive and negative image pairs
- Verification threshold system to determine identity match
- Support for collecting anchor, positive, and negative training samples

## Project Structure

```
face-recognition-deep-learning/
├── application_data/
│   ├── input_image/        # Storage for current verification attempt
│   └── verification_images/ # Reference images for verification
├── data/
│   ├── anchor/            # Training images of the target person
│   ├── positive/          # Additional images of the target person
│   └── negative/          # Images of other people for training
├── training_checkpoints/  # Model checkpoints saved during training
├── .venv/                 # Python virtual environment
├── siamesemodel.h5        # Trained model file
└── main.py                # Main application code
```

## Technologies Used

- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib

## How It Works

### Siamese Network Architecture

The system uses a Siamese Neural Network architecture which consists of:
1. An embedding network that processes each image
2. A distance layer that computes the L1 distance between embeddings
3. A classification layer that determines if the images match

### Training Process

The model is trained on pairs of images:
- Positive pairs: Images of the same person
- Negative pairs: Images of different people

The model learns to output a similarity score closer to 1 for matching faces and closer to 0 for non-matching faces.

### Verification System

The verification process:
1. Captures an input image from the webcam
2. Compares it against multiple verification images
3. Uses detection and verification thresholds to determine identity

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-recognition-deep-learning.git
cd face-recognition-deep-learning
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install tensorflow opencv-python numpy matplotlib
```

## Usage

### Creating Training Data

To collect your own training data, uncomment the webcam data collection section in the code and run:

```bash
python main.py
```

- Press 'a' to capture anchor images (images of yourself)
- Press 'p' to capture positive images (additional images of yourself)
- Press 'q' to quit the data collection process

### Training the Model

To train the model on your data:

1. Ensure you have collected anchor, positive, and negative images
2. Uncomment the training section in the code
3. Run:
```bash
python main.py
```

### Running Verification

To run the face verification system:

1. Ensure you have verification images in the `application_data/verification_images` folder
2. Run:
```bash
python main.py
```

3. Press 'v' to capture an input image and verify your identity
4. Press 'q' to quit

## Model Performance

The model uses precision and recall metrics to evaluate performance. The verification thresholds can be adjusted based on the required security level:

- Higher detection threshold: More security but possible false negatives
- Lower detection threshold: More convenience but possible false positives

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
