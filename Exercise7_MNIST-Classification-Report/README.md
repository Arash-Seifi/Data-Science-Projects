# MNIST Classification Using a Multilayer Perceptron (MLP)

## Introduction
This report outlines the process of building, training, and evaluating a Multilayer Perceptron (MLP) model for classifying handwritten digits from the MNIST dataset. The MNIST dataset consists of 28x28 grayscale images of digits (0-9) and is widely used for benchmarking machine learning models.
![Pred2](https://github.com/user-attachments/assets/db4e22f5-ac47-41d7-8aa2-dca6348f5e68)

## Dataset Preparation
The MNIST dataset was loaded and preprocessed as follows:
- Normalization: Pixel values were scaled to the range [0, 1].
- One-hot Encoding: Labels were converted to one-hot encoded vectors.
- Splitting: The training data was divided into training and validation sets.

## Model Architecture
The MLP model was designed with the following layers:
1. **Flatten Layer**: Converts the 28x28 input image into a 1D vector.
2. **Dense Layer (128 neurons)**: First hidden layer with ReLU activation.
3. **Dense Layer (64 neurons)**: Second hidden layer with ReLU activation.
4. **Dense Layer (10 neurons)**: Output layer with softmax activation for multi-class classification.

## Model Compilation
The model was compiled using:
- **Optimizer**: Adam optimizer.
- **Loss Function**: Categorical cross-entropy.
- **Metrics**: Accuracy.

## Training
The model was trained for 20 epochs with a batch size of 32. Training and validation accuracy/loss were recorded for each epoch.

## Evaluation
The model was evaluated on the test dataset, achieving a test accuracy of [insert accuracy here].

## Visualization
### Accuracy and Loss Plots
- **Accuracy Plot**: Shows the training and validation accuracy over epochs.
- **Loss Plot**: Shows the training and validation loss over epochs.

### Sample Predictions
Random samples from the test set were selected, and the model's predictions were visualized alongside the actual images.

## Conclusion
The MLP model demonstrated effective performance on the MNIST dataset, achieving high accuracy on both the validation and test sets. Further improvements could be explored by tuning hyperparameters or experimenting with more complex architectures.
