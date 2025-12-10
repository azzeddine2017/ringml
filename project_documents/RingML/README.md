# RingML: High-Performance Machine Learning Library for Ring

## 1. Vision
To build a modular, high-performance Deep Learning library in Ring language, leveraging the `FastPro` extension for accelerated matrix operations. The goal is to provide a syntax similar to PyTorch/Keras for building Neural Networks and eventually Language Models.

## 2. Technical Stack
- **Language:** Ring
- **Core Extension:** `FastPro` (Mandatory for Matrix/Vector ops)
- **Architecture:** OOP (Classes for Tensors, Layers, Models)

## 3. Architecture & Modules

The project is divided into 3 main phases (Modules).

### Module 1: The Core (Tensor Engine)
Wrapping `FastPro` functions into a usable `Tensor` class to handle shapes, dimensions, and math operations intuitively.

- [x] **T01.01**: Create `RingML` package and directory structure.
- [x] **T01.02**: Implement `Tensor` class foundation (Init, Random, Zeros).
- [x] **T01.03**: Implement Math Operations (Add, Sub, Mul, DotProduct) using `FastPro`.
- [x] **T01.04**: Implement Mathematical Transformations (Sigmoid, ReLU, Transpose).

### Module 2: Neural Building Blocks (Layers)
Building the layers required for a neural network using the `Tensor` class.

- [x] **T02.01**: Define abstract `Layer` class (Forward/Backward signatures).
- [x] **T02.02**: Implement `Dense` (Linear) Layer (Weights, Biases, Forward pass).
- [x] **T02.03**: Implement Activation Layers (ReLU, Sigmoid, Softmax).
- [x] **T02.04**: Implement Loss Functions (MSE, CrossEntropy).

### Module 3: Model & Optimization
Putting it all together to train a network.

- [x] **T03.01**: Implement `Sequential` model container.
- [x] **T03.02**: Implement `Optimizer` class (SGD - Stochastic Gradient Descent).
- [x] **T03.03**: Implement Backpropagation logic (Derivatives calculation).
- [x] **T03.04**: Create a standard "XOR" or "MNIST" example to verify the library.

## 4. Constraints & Guidelines
- **FastPro Only:** Do not write loop-based math loops in Ring; strictly use `updateList` with `:matrix` commands for performance.
- **Matrix Representation:** Input data should be treated as 2D lists (Matrices).
- **Naming:** Use standard ML terminology (Weights, Bias, Gradients).

## 5. Documentation
- Each module must have a corresponding `.readme.md` file.
- Code must be commented with inputs/outputs.





# RingML: Deep Learning Library for Ring Language

RingML is a lightweight, modular Deep Learning framework built from scratch in Ring. It leverages the **FastPro** extension for accelerated matrix operations, providing a PyTorch-like API for building and training Neural Networks.

## 🚀 Features

*   **Tensor Engine**: Wraps `FastPro` C-extension for matrix math (MatMul, Transpose, etc.).
*   **Automatic Differentiation**: Implements full Backpropagation (Backward Pass).
*   **Layers**:
    *   `Dense` (Fully Connected).
    *   `Sigmoid`, `ReLU`, `Tanh` (Activations).
*   **Optimization**:
    *   `SGD` (Stochastic Gradient Descent).
    *   `MSELoss` (Mean Squared Error).
*   **API**:
    *   `Sequential` model container for easy stacking of layers.

## 📦 Project Structure

```text
src/
├── core/
│   └── tensor.ring       # The mathematical heart (Matrix ops)
├── layers/
│   ├── layer.ring        # Abstract Base Class
│   ├── dense.ring        # Fully Connected Layer
│   └── activation.ring   # Activation Functions
├── loss/
│   └── mse.ring          # Mean Squared Error
├── model/
│   └── sequential.ring   # Model Container
├── optim/
│   └── sgd.ring          # Optimizer
└── ringml.ring           # Main Loader
examples/
└── xor_train.ring        # Proof-of-concept (XOR Problem)
```
## ⚡ Quick Start

### Solving XOR Problem

```Ring
load "src/ringml.ring"

# 1. Prepare Data
inputs  = new Tensor(4, 2) { aData = [[0,0], [0,1], [1,0], [1,1]] }
targets = new Tensor(4, 1) { aData = [[0],   [1],   [1],   [0]]   }

# 2. Build Model
model = new Sequential
model.add(new Dense(2, 4)) # Input: 2, Hidden: 4
model.add(new Sigmoid)
model.add(new Dense(4, 1)) # Output: 1
model.add(new Sigmoid)

# 3. Setup Training
optimizer = new SGD(0.5)   # Learning Rate
criterion = new MSELoss

# 4. Training Loop
for epoch = 1 to 5000
    # Forward
    preds = model.forward(inputs)
    
    # Backward
    lossGrad = criterion.backward(preds, targets)
    model.backward(lossGrad)
    
    # Update
    for layer in model.getLayers()
        optimizer.update(layer)
    next
next
```
# 5. Predict
model.forward(inputs).print()

**🛠 Dependencies**
*  Ring Language (1.24 or later)
*  FastPro Extension (Must be loaded/dll available)

## 📝 License
Open Source. 

