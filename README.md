# Transfer Learning with CIFAR-10: A Two-Stage Approach

This repository contains a Jupyter Notebook that demonstrates a two-stage transfer learning approach using the CIFAR-10 dataset. The process is divided into two stages: initial training on a subset of classes and fine-tuning the model for the remaining classes.



## Introduction
Transfer learning is a powerful technique in deep learning where a model developed for a specific task is reused as the starting point for a model on a second task. This notebook demonstrates how to apply transfer learning to the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Dataset
The CIFAR-10 dataset is used in this project. It contains 50,000 training images and 10,000 test images, divided into 10 classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Methodology

### Stage 1: Initial Training on a Subset of Classes
In the first stage, a convolutional neural network (CNN) is trained on a subset of the CIFAR-10 classes. The selected classes for initial training are:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog

#### Steps:
1. **Import Necessary Libraries**: TensorFlow, Keras, NumPy, and other required libraries are imported.
2. **Load and Preprocess the CIFAR-10 Dataset**: The dataset is loaded and normalized. The selected classes are filtered, and labels are converted to categorical format.
3. **Build a CNN Model**: A CNN model is designed with convolutional layers, max-pooling layers, and fully connected layers.
4. **Train the Model**: The model is compiled with an appropriate loss function, optimizer, and evaluation metrics. It is then trained using the training data and validated using the test data.

### Stage 2: Fine-Tuning the Model for Remaining Classes
In the second stage, the model is fine-tuned for the remaining classes:
- Horse
- Ship
- Truck

#### Steps:
1. **Filter Data for Remaining Classes**: The data for the remaining classes is filtered.
2. **Fine-Tune the Model**: The pre-trained model from Stage 1 is fine-tuned using the data from the remaining classes.
3. **Evaluate the Model**: The fine-tuned model is evaluated on the test data.

## Findings

### Comparison of Stage 1 and Stage 2
- **Accuracy**: 
  - Stage 1 Accuracy: 0.7310
  - Stage 2 Accuracy: 0.8383
  - The accuracy of the model improved significantly from Stage 1 to Stage 2, indicating that fine-tuning the model on additional classes helped it generalize better.

- **Precision and Recall**: 
  - Stage 1 Precision: 0.7756, Recall: 0.6806
  - Stage 2 Precision: 0.8688, Recall: 0.7943
  - There was an observable increase in both precision and recall metrics, suggesting that the model became more effective at correctly identifying the classes it was trained on.

- **F1 Score**: 
  - Stage 1 F1 Score: 0.7217
  - Stage 2 F1 Score: 0.8374
  - The F1 score showed improvement, reflecting a better balance between precision and recall, which is crucial for multi-class classification tasks.

### Observations on Transfer Learning
- **Generalization**: The model demonstrated improved generalization capabilities, as evidenced by the enhanced performance metrics in Stage 2.
- **Feature Retention**: By freezing the layers of the pre-trained model, we retained valuable features learned from the initial training, which contributed to better performance on the new classes.
- **Faster Training**: Transfer learning allowed the model to leverage previously learned features, resulting in faster training times and reduced computational resources needed for Stage 2 compared to training a model from scratch.
