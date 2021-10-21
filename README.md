# ML-Physicists-Examples

This repository contains worked examples that I found useful when following the Max Planck Institute course 'Machine Learning for Physicists' https://machine-learning-for-physicists.org/

# Examples

The [ExampleNotebooks](./ExampleNotebooks) folder contains several files (.py converted from .ipynb files) that illustrate several examples of convolutional neural networks. 

- The [CNNTraining](./ExampleNotebooks/CNNTraining.py) file contains a simple example for training a convolutional network to recognise shapes (circles and squares)

- The [DenoiseCNN](./ExampleNotebooks/DenoiseCNN.py) file contains a denoising auto-encoder, using the same shape recognition network from [CNNTraining](./ExampleNotebooks/CNNTraining.py)

- The [DigitRecognition](./ExampleNotebooks/DigitRecognition.py) contains an example network that recognises digits using the MNIST handwritten digits database. 

# Written code

The [WrittenExamples](./WrittenExamples) folder contains python files that I wrote to practice implementing simple neural networks from scratch (no libraries). 

- The [IntroNetwork](./WrittenExamples/IntroNetwork.py) file contains a manual implementation of a simple sequential network with three hidden layers.

- The [BackpropNetwork](./WrittenExamples/BackpropNetwork.py) file implements backpropagation to the [IntroNetwork](./WrittenExamples/IntroNetwork.py).
