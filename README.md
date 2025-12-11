# 🧠 RingML: Deep Learning Library for Ring

**RingML** is a lightweight, modular, and extensible Deep Learning framework written in [Ring](https://ring-lang.net). It provides a PyTorch-like API for building, training, and deploying Neural Networks, powered by the **FastPro** C-extension for high-performance matrix operations.

---

## 🚀 Features

*   **Tensor Engine**: fast matrix operations (MatMul, Transpose, Broadcasting).
*   **Layer-Based Architecture**: Modular `Sequential` models with `Dense`, `ReLU`, `Sigmoid`, `Softmax`.
*   **Automatic Differentiation**: Full implementation of Backpropagation.
*   **Optimizers & Loss**: `SGD` optimizer, `MSELoss` for regression, `CrossEntropyLoss` for classification.
*   **Data Handling**: `Dataset` and `DataLoader` for mini-batch processing.
*   **Model Persistence**: Save and Load trained models easily.

---

## 📦 Installation

1.  Ensure you have **Ring 1.20+**.
2.  Ensure the **FastPro** extension is available (dll/so).
3.  Clone this repository:
    ```bash
    git clone https://github.com/yourusername/RingML.git
    ```

---

## ⚡ Quick Start

### 1. Classification (XOR Problem)

```ring
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
optimizer = new SGD(0.5)
criterion = new MSELoss

# 4. Train
for epoch = 1 to 5000
    preds = model.forward(inputs)
    loss  = criterion.forward(preds, targets)
    
    # Backpropagation
    grad = criterion.backward(preds, targets)
    model.backward(grad)
    
    # Update Weights
    for layer in model.getLayers() optimizer.update(layer) next
next

model.forward(inputs).print()
```

### 2. Multi-Class Classification (Softmax)

```ring
load "src/ringml.ring"

# Model for 3 classes
model = new Sequential
model.add(new Dense(10, 20)) 
model.add(new Sigmoid)
model.add(new Dense(20, 3)) 
model.add(new Softmax)       # Output Probabilities

criterion = new CrossEntropyLoss

# 4. Train
for epoch = 1 to 5000
    preds = model.forward(inputs)
    loss  = criterion.forward(preds, targets)
    
    # Backpropagation
    grad = criterion.backward(preds, targets)
    model.backward(grad)
    
    # Update Weights
    for layer in model.getLayers() optimizer.update(layer) next
next

model.forward(inputs).print()
```

### 3. Save & Load Models

```ring
load "src/ringml.ring"

# Save
model.saveWeights("mymodel.rdata")

# Load
model2 = new Sequential
# ... (Define same architecture) ...
model2.loadWeights("mymodel.rdata")
```

---

## 📂 Project Structure

```text
RingML/
├── src/
│   ├── core/           # Tensor & Math Logic
│   ├── data/           # Dataset & DataLoader
│   ├── layers/         # Dense, Activations, Softmax
│   ├── loss/           # MSE, CrossEntropy
│   ├── model/          # Sequential Container
│   ├── optim/          # Optimizers (SGD)
│   └── ringml.ring     # Library Loader
├── examples/           # Ready-to-run demos
└── README.md           # Documentation
```

---

## 🛠 Status & Performance

Current Version: 1.0
Performance: Uses Ring loops for some operations (Safe Mode) pending specific updates to FastPro C-Extension (Double precision support).

---

## 📄 License

Open Source under MIT License.
