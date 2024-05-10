# Leprosy_Detection
A model that integrates SVM with CNN for the discriminating between Leprosy and Non-Leprosy 

This is a convolutional neural network model designed for binary classification of leprosy detection from images. It utilizes a pre-trained EfficientNet model from TensorFlow Hub as a powerful feature extractor, leveraging the weights learned from a large dataset. The extracted features are then passed through additional dense layers to learn task-specific representations. Data augmentation techniques like rescaling, zooming, rotation, and brightness adjustment are applied to the training data to increase diversity and improve generalization. The model is trained with the hinge loss function, suitable for binary classification tasks, and the Adam optimizer. Regularization techniques like L1L2 regularization and early stopping are employed to prevent overfitting and ensure optimal performance on the validation data.

