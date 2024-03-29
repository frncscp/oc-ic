# One-Class Image Classification with AI Algorithms

This research aims to find the most optimal way to perform image classification where just one type of class is important. The following algotithms are used:

- *Convolutional Neural Networks (Mid to Very Good perfomance, fast)*
- *Classic Machine Learning Algorithms (Mid to Good performance, fast)*
- *Visual Transformers (Great perfromance, medium-to-fast)*
- *Zero-Shot Classification (Great performance, medium-to-fast)*

## Getting started
To be able to use the training and analysis code, you will need to meet the following requirements:
- Use .h5 Tensorflow/Keras (training/testing) or HuggingFace (testing) models.
- Have a training dataset consisting of a positive and a pseudo-negative class, divided just on classes, not on subsets.

- You should have two folders for your models:
     - One for models that are going to have their weights reinitialized randomly. 
     - The other for models where their weights matter and you want to fine-tune them.
       
  _Note that for training, this code only accepts Tensorflow/Keras models, but you can use already trained Hugging Face models for testing (you'll probably need to sign in first)._

Then you can execute training_and_analysis.py! You'll have to answer some questions to gather the data needed for the computation.
