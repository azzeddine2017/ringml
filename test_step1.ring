load "src/ringml.ring"

see "=== RingML Core Tensor Tests ===" + nl

# 1. Initialization
see "1. Initialization Tests:" + nl
t1 = new Tensor(2, 2) { fill(1) }
t2 = new Tensor(2, 2) { fill(2) }
see "   T1 (Filled with 1):" 
t1.print()

# 2. Element-wise Addition
see "2. Addition Test (T1 + T2):" + nl
t1.add(t2) 
t1.print()

# 3. Scalar Multiplication
see "3. Scalar Multiplication (T1 * 2):" + nl
t1.scalar_mul(2) 
t1.print()

# 4. Matrix Multiplication (MatMul)
see "4. Matrix Multiplication Test:" + nl
matA = new Tensor(1, 2) { 
    aData = [[1, 2]] 
}
matB = new Tensor(2, 1) { 
    aData = [[3], [4]] 
}

see "   MatA (1x2): " + nl see matA.aData see nl
see "   MatB (2x1): " + nl see matB.aData see nl

# (1*3 + 2*4) = 3 + 8 = 11
# Use matmul for (M x N) * (N x P)
res = matA.matmul(matB)
see "   Result (Should be [[11]]):" + nl
res.print()

# 5. Transpose
see "5. Transpose Test:" + nl
transT = new Tensor(2, 1) { 
    aData = [[1],[2]]
}
see "   Original (2x1):" + nl transT.print()
transT.transpose()
see "   Transposed (1x2):" + nl transT.print()

see "=== All Tests Completed ===" + nl