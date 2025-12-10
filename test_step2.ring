load "src/ringml.ring"

see "=== RingML Layers Test (Retry) ===" + nl

# 1. Define Input Data (Batch Size = 1, Features = 3)
see "1. Creating Input (1x3)..." + nl
input = new Tensor(1, 3)
input.aData = [[1, 0.5, -1]]
input.print()

# 2. Define Dense Layer (Input=3, Output=2)
see "2. Initializing Dense Layer (3 inputs -> 2 outputs)..." + nl
# This calls scalar_mul(0.01) internally, which should now work safely
layer1 = new Dense(3, 2)

see "   Weights (3x2) [Should be small randoms]:" + nl 
layer1.oWeights.print()
see "   Biases (1x2) [Should be zeros]:" + nl 
layer1.oBias.print()

# 3. Forward Pass
see "3. Performing Forward Pass..." + nl
# This calls matmul() which now uses FastPro Case 406
output = layer1.forward(input)

see "   Output (1x2):" + nl
output.print()

# 4. Activation Test (Sigmoid)
see "4. Applying Sigmoid Activation..." + nl
act = new Sigmoid
output = act.forward(output)
see "   After Sigmoid:" + nl
output.print()

see "=== Layers Test Completed ===" + nl