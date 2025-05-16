This project implements an emotion detection model that classifies tweets into emotion categories using a Bidirectional LSTM neural network.
The model is trained on the emotion dataset from Hugging Face's nlp library (now datasets), with preprocessing, training, evaluation, and visualization workflows built using TensorFlow, NumPy, and Matplotlib.

The code performs the following:
Loads and processes the Emotion NLP dataset
Text tokenization using Keras Tokenizer with padding and truncation
Sequence modeling with an Embedding layer + Bidirectional LSTM
Multi-class classification using softmax output (6 emotion classes)
Training visualization of loss and accuracy
Confusion matrix for model performance insight
Predicts emotions from randomly selected test tweets

The model detects the following 6 emotion categories (order may vary depending on dataset version):
Joy
Sadness
Anger
Fear
Love
Surprise

