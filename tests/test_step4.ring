load "../src/ringml.ring"
decimals(8)

see "=== RingML Single Step Training Test ===" + nl

# 1. Data (Inputs and Targets)
input  = new Tensor(1, 2) { aData = [[0.5, 0.1]] }
target = new Tensor(1, 1) { aData = [[0.7]] }

# 2. Model (Linear Layer)
layer = new Dense(2, 1)
# Force weights to known values for debugging
layer.oWeights.fill(0.5) 
layer.oBias.fill(0.0)

# 3. Setup Components
criterion = new MSELoss
optimizer = new SGD(0.1) # Learning Rate 0.1

see "--- Before Update ---" + nl
# Forward
pred = layer.forward(input)
loss = criterion.forward(pred, target)
see "Prediction: " pred.print()
see "Target:     " target.print()
see "Loss: " + loss + nl

# Backward
lossGrad = criterion.backward(pred, target)
layer.backward(lossGrad)

# Update
optimizer.update(layer)

see "--- After Update ---" + nl
# Forward again to check improvement
predNew = layer.forward(input)
lossNew = criterion.forward(predNew, target)

see "New Prediction: " predNew.print()
see "New Loss: " + lossNew + nl

if lossNew < loss
    see "SUCCESS: Loss decreased! The model is learning." + nl
else
    see "FAILURE: Loss did not decrease." + nl
ok