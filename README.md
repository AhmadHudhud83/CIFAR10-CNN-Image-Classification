# CIFAR-10 Image Classification with CNN

## Introduction
The **CIFAR-10 dataset** is a widely recognized computer vision dataset used for object recognition. It is a subset of the 80 million tiny images dataset, collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. The dataset comprises **60,000 32x32 color images** across **10 object classes**, with **6,000 images per class**. The goal of this project is to develop a convolutional neural network (CNN) model to accurately classify images from this dataset into one of the 10 classes.

## Dataset
- **Source**: Imported from the TensorFlow Datasets library as a Tensor Data object.
- **Splits**:
  - **Training Set**: 40,000 samples
  - **Validation Set**: 10,000 samples
  - **Test Set**: 10,000 samples
  - **Total Size**: 60,000 samples
- **Exploration**: Sample images were visualized to confirm their dimensions (**32x32 pixels**) and **RGB channels**, providing insight into the dataset's characteristics.

## Preprocessing
- Preprocessing steps were seamlessly integrated into the pipeline and applied to all datasets (training, validation, and test sets).
- **Data Augmentation**: Exclusively applied to the training set to enhance model generalization by introducing variability, such as random flips or rotations.

## Model Architecture
- **Initial Model**: A basic CNN was first implemented, achieving approximately **65% validation accuracy**. However, it struggled to capture complex patterns and exhibited overfitting.
- **Evolution**:
  - Complexity was increased by adding more hidden layers, incorporating **data augmentation**, **batch normalization**, and **dropout**. This yielded a modest improvement (~3%).
  - A significant breakthrough came with the adoption of **two convolutional layers before each max-pooling layer**, enabling the model to learn intricate patterns more effectively.
- **Best Model**: Combined the two-conv-layers-per-max-pooling approach with batch normalization and dropout for optimal performance.

## Training
- **Callbacks**:
  - **Learning Rate Scheduler**: Dynamically reduced the learning rate (starting at 1e-4) to improve convergence.
  - **Early Stopping**: Halted training if the validation performance plateaued, preventing overfitting.
- **Monitoring**: Accuracy and loss charts were generated for both training and validation sets to track progress.
- **Process**: The validation set was reserved for tuning, while the test set was used for final evaluation.

## Results
After training, the model was saved, reloaded, and evaluated. The table below summarizes the performance of various models tested:

| Model                                              | Accuracy | Loss   | Val Accuracy | Val Loss | Final Learning Rate |
|----------------------------------------------------|----------|--------|--------------|----------|---------------------|
| Basic Model                                        | 0.6395   | 1.0290 | 0.6524       | 0.9913   | 1.0e-04             |
| Complex Model (False Augmentation)                 | 0.6480   | 1.0139 | 0.6690       | 0.9149   | 1.0e-07             |
| More Complex Model (Augmentation, Batch Norm, Dropout) | 0.7964   | 0.5829 | 0.8377       | 0.4587   | 1.0e-08             |
| Same as Above but No Batch/Dropout                 | 0.7261   | 0.7720 | 0.7305       | 0.7599   | 1.0e-10             |
| Best Model (2 Conv Layers per Pooling, Batch/Dropout) | 0.8136   | 0.5316 | **0.8521**   | 0.4283   | 1.0e-07             |

- **Best Model Performance**:
  - **Validation Accuracy**: 85.21%
  - **Test Accuracy**: 80.52%
- *Note*: Learning rates represent the final values after dynamic reduction by the scheduler.

## Discussion
- **Initial Model**: The basic CNN was underpowered and prone to overfitting, limiting its ability to generalize.
- **Incremental Improvements**: Doubling hidden layers and adding data augmentation, batch normalization, and dropout improved performance slightly (~3%).
- **Key Insight**: Using **two convolutional layers before each max-pooling layer** markedly enhanced learning, boosting validation accuracy to 83% within the first 10 epochs.
- **Ablation Studies**:
  - Removing batch normalization and dropout reduced performance by up to **10%**, underscoring their importance.
  - Data augmentation mitigated overfitting but did not directly increase accuracy.
- **Final Tuning**: Fine-tuning layer parameters (especially batch normalization and dropout) and increasing the number of epochs maximized performance.

## Conclusion
Through iterative refinement, the final CNN model achieved a **validation accuracy of 85.21%** and a **test accuracy of 80.52%**. The use of two convolutional layers per max-pooling layer, combined with regularization techniques, proved pivotal. Future enhancements could involve exploring deeper architectures or leveraging transfer learning for even better results.

## Additional Analysis
- **Confusion Matrix**: Employed for advanced evaluation, revealing classification overlaps and errors per class.
- **Filter Visualization**: Filters from the first layer and feature maps across layers were analyzed to understand the model's learned representations.

---

**Author**: Ahmad Hudhud - 12220718
