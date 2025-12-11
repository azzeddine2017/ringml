load "../src/ringml.ring"

see "=== Sign Logic Diagnosis ===" + nl

# 1. Test Scalar Mul with Negative
see "1. Testing Scalar Mul (-1 * 5):" + nl
t1 = new Tensor(1,1) { aData=[[5.0]] }
t1.scalar_mul(-1.0)
t1.print() 
# المتوقع: -5.0

# 2. Test MatMul with Negative
see "2. Testing MatMul (Positive * Negative):" + nl
# [1] * [-1] = [-1]
mA = new Tensor(1,1) { aData=[[1.0]] }
mB = new Tensor(1,1) { aData=[[-1.0]] }
mRes = mA.matmul(mB)
mRes.print()
# المتوقع: -1.0

# 3. Test SGD Step Logic manually
see "3. Simulating SGD Step:" + nl
# Weight = 0.5
# Grad = -0.1 (Negative gradient means we want to increase weight)
# LR = 1.0
# Step = Grad * LR = -0.1
# NewWeight = Weight - Step = 0.5 - (-0.1) = 0.6 (Should Increase)

w = new Tensor(1,1) { aData=[[0.5]] }
stepe = new Tensor(1,1) { aData=[[-0.1]] } # Negative Step
w.sub(stepe) # 0.5 - (-0.1)
see "Result (Should be 0.6): "
w.print()