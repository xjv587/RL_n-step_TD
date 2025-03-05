# Implementing n-step TD with function approximation
This project focuses on implementing the n-step semi-gradient TD(0) algorithm using two function approximation methods: linear approximation with tile coding and neural networks. The goal is to evaluate and improve policies in a continuous environment using the OpenAI Gym library, specifically the "MountainCar-v0" environment.

- Implementing the **n-step semi-gradient TD(0) algorithm** to estimate value functions.

- **Tile Coding:** Developing a linear function approximation using tile coding, ensuring complete coverage of the state space and handling floating-point precision issues.

- **Neural Networks:** Building a neural network with PyTorch to approximate value functions, using a specific architecture with four layers and ReLU activations, and optimizing with Adam.
