# Deep Feed-Forward Neural Network for ECG Classification

This project implements a **Deep Feed-Forward Neural Network (DFFN)** using TensorFlow and Keras to classify ECG data into two categories: Normal and Abnormal. The task involves building a neural network from scratch, preprocessing the data, and experimenting with different network architectures and hyperparameters.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to build and train a deep feed-forward neural network to classify ECG data into two classes: **Normal** and **Abnormal**. The network architecture and hyperparameters are chosen carefully to optimize performance, and the model is evaluated using metrics such as accuracy, training loss, and validation accuracy.

## Dataset

The dataset used is an ECG dataset, which can be downloaded from the following source:
- [ECG Database](https://www.cs.cmu.edu/~bobski/data/)

### Preprocessing:
- Only the first 75 measurements from each ECG recording are used. Any recordings shorter than 75 are either padded with zeros or the average values.

## Methodology

### Network Architecture

The deep feed-forward neural network was built using **TensorFlow** and **Keras** libraries. The model architecture consists of:

1. **Input Layer**: Accepts a feature vector of size 150 (padded or truncated ECG measurements).
2. **Hidden Layers**:
   - First hidden layer: 64 neurons, ReLU activation.
   - Second hidden layer: 32 neurons, ReLU activation.
3. **Output Layer**: A single neuron with a **Sigmoid activation function** to classify Normal vs. Abnormal ECG.

### Training and Hyperparameters:

- **Optimizer**: Adam optimizer with a learning rate of 0.001 (tested with 0.0001 but performed better with 0.001).
- **Loss Function**: Binary Cross-Entropy.
- **Epochs**: The model is trained for 100 epochs, though the model converged around 30 epochs, with further improvements seen up to 80 epochs.
- **Training/Validation Split**: 80/20 split with a random seed for reproducibility.

### Evaluation Metrics:
- **Training Loss**
- **Validation Loss**
- **Training Accuracy**
- **Validation Accuracy**

## Results

The model achieved approximately **80% accuracy** in classifying Normal and Abnormal ECG data. The following graphs represent the training process:
- **Figure 1**: Training Loss & Validation Loss
- **Figure 2**: Training Accuracy & Validation Accuracy

The model converged after around 30 epochs, but training was continued for 100 epochs to observe stability.

## Conclusion

This deep feed-forward neural network achieved a reasonable accuracy of around 80%, indicating that the architecture and preprocessing steps were effective. However, improvements could be made by increasing the dataset size or fine-tuning hyperparameters further for better results.

## Installation

1. Clone the repository:
   

2. Install the required dependencies:
   

3. Download the **ECG dataset** and place it for example in the `data/` directory:
    ```bash
    data/
        |-- ecg_data/
    ```

## Usage

1. **Train the DFFN model**:
   

2. **Evaluate the model**:
    The training and validation loss, as well as accuracy, will be plotted to monitor the performance over the epochs.


