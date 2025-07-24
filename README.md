# Intel Image Classification
![build](https://img.shields.io/badge/build-passing-brightgreen)
![project-type](https://img.shields.io/badge/project_type-image--classification-orange)
![python](https://img.shields.io/badge/python-3.12.10-blue)

This project performs **image classification** on the
[Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data)
using a custom made Convolutional Neural Network (CNN) model.

## Project Purpose
- Learn how to build a neural network using PyTorch's nn.modules.
- Understand how to implement loss function, optimisation and scheduler in training loop.
- Perform image classification of natural scenes using raw image data.


## Workflow
1. **Data Preparation**
   - Reading of image files
   - Resizing and conversion to tensors using Torchvision's transforms
   - Loading of data using Torchvision's ImageFolder and PyTorch's DataLoader
2. **Model Architecture**
   - PyTorch's nn.Module as base class
   - Dynamic number of convolution layers
   - Max pooling layer. Pooling performed after every convolution layer
   - ReLU activation function
   - Flattening of last pooling layer
   - Fully connected layers
   - Make use of working modules provided in PyTorch
3. **Training loop**
   - Loss function: Cross Entropy Loss
   - Optimiser: AdamW
   - Scheduler: Cosine Annealing
   - 25 epoch
   - Utilise GPU if available 
4. **Evaluation**
   - Model evaluated on validation dataset
   - Metrics used: accuracy, precision, recall, and F1-score

## Tools & Libraries
- Pillow - For reading image files
- PyTorch - For building of model and training loop
- Torchvision - For transformation and loading of image files
- Scikit-learn (sklearn) - For evaluation metrics

## Model Performance
Evaluated on the validation dataset, using the best epoch (Epoch 23).

|  Metric  | Score  |
|----------|--------|
| Accuracy | 82.77% |
| Precision| 82.71% |
| Recall   | 82.77% |
| F1-score | 82.73% |
