load "../src/ringml.ring"
decimals(8)

see "=== RingML Backpropagation Test ===" + nl

# 1. Setup Data
see "1. Input Data..." + nl
input = new Tensor(1, 3)
input.aData = [[0.5, 0.1, -0.2]]
input.print()

# 2. Setup Layer
see "2. Dense Layer & Forward..." + nl
layer = new Dense(3, 2)
output = layer.forward(input)
output.print()

# 3. Simulate Gradient from next layer (Mock Error)
see "3. Simulating Gradient from Loss..." + nl
gradOutput = new Tensor(1, 2)
gradOutput.fill(0.1) # Suppose we have small error
gradOutput.print()

# 4. Backward Pass (Dense)
see "4. Dense Backward..." + nl
gradInput = layer.backward(gradOutput)

see "   Gradient w.r.t Input (Passed to prev layer):" + nl
gradInput.print()

see "   Gradient w.r.t Weights (Stored in layer):" + nl
layer.oGradWeights.print()

see "   Gradient w.r.t Bias (Stored in layer):" + nl
layer.oGradBias.print()

# 5. Sigmoid Backward
see "5. Sigmoid Backward..." + nl
act = new Sigmoid
# Forward first to cache output
actOut = act.forward(output)
# Backward
gradAct = act.backward(gradOutput) # Using same dummy grad
see "   Sigmoid Gradient:" + nl
gradAct.print()

see "=== Backprop Test Completed ===" + nl