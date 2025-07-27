# BareCNN: A Bare-Bones Convolutional Neural Network Framework

## 🌟 Overview

BareCNN is a minimalist, from-scratch implementation of a Convolutional Neural Network (CNN) framework, built entirely using NumPy. The primary goal of this project is to provide a clear, understandable, and functional deep learning library without relying on high-level frameworks like PyTorch or TensorFlow. It's designed for educational purposes, allowing users to delve into the core mechanics of CNNs, including forward and backward propagation, loss calculation, and optimization.

This project represents **Version 0.1.0**, a fully functional initial release capable of defining, training, and evaluating basic CNN architectures.

## ✨ Features

BareCNN provides the fundamental building blocks necessary to construct and train deep neural networks:

### Core Components:

- **Parameter**: Manages trainable weights and biases with their gradients.
- **Math**: Essential numerical operations, including efficient im2col and col2im for convolutional layers.
- **ParamInit**: Various weight and bias initialization strategies (He, Xavier, Zeros, Small Positive).

### Layer Implementations:

- **ConvolutionalLayer**: Core building block for feature extraction.
- **PoolingLayer**: Supports Average Pooling for spatial downsampling.
- **Flatten**: Transforms multi-dimensional feature maps into 1D vectors.
- **LinearLayer**: Standard fully connected layers.
- **Activation Functions**: ReLU, Sigmoid, TanH, and Softmax (for inference).

### Loss Functions:

- **MSELoss**: Mean Squared Error for regression tasks.
- **SoftmaxCrossEntropyLoss**: For multi-class classification, including automatic class weighting based on training data frequency.

### Optimizers:

- **SGD**: Stochastic Gradient Descent with optional Momentum.
- **Adam**: Adaptive Moment Estimation optimizer.

### Model Management:

- **Model class**: Orchestrates the entire network, managing layers, loss, optimizer, and the training loop.
- **Model Persistence**: Ability to save and load trained model parameters (`.npz` files).

### Evaluation Metrics:

- **Accuracy**: Overall classification correctness.
- **Precision**: Macro-averaged precision.
- **Recall**: Macro-averaged recall.

## 🚀 Installation

BareCNN is designed as a local Python package. To install it in your environment:

```bash
# Clone the repository:
git clone https://github.com/exiort/BareCNN.git  # Replace with your actual repo URL
cd BareCNN

# Create and activate a virtual environment (recommended):
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install BareCNN library:
pip install BareCnn/

# Verify installation:
python -c "import barecnn; from barecnn.models.model import Model; print('BareCNN installed successfully!')"
```

## 🤝 Contributing

Contributions are welcome! Open an issue or PR on GitHub.

## 📄 License

MIT License. See the LICENSE file.

## 🙏 Acknowledgements

- Inspired by LeNet-5 by Yann LeCun et al.
- Thanks to the NumPy community.
