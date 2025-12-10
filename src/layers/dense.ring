# File: src/layers/dense.ring
# Description: Fully Connected (Dense) Layer with Forward & Backward
# Author: Code Gear-1


class Dense from Layer
    oWeights        # Tensor (nInput, nNeurons)
    oBias           # Tensor (1, nNeurons)
    oInput          # Cache input for backprop
    
    oGradWeights    # Gradient w.r.t Weights
    oGradBias       # Gradient w.r.t Bias

    nInputSize
    nNeurons

    func init nIn, nOut
        nInputSize = nIn
        nNeurons   = nOut
        
        # Initialize Weights
        oWeights = new Tensor(nInputSize, nNeurons)
        
        # IMPROVED INITIALIZATION (Uniform Random between -1 and 1)
        # This breaks symmetry and allows learning XOR effectively.
        for r = 1 to nInputSize
            for c = 1 to nNeurons
                # random(2000) gives 0..2000. 
                # Divide by 1000 -> 0..2. 
                # Subtract 1 -> -1..1
                val = (random(2000) / 1000.0) - 1.0
                oWeights.aData[r][c] = val
            next
        next

        # Initialize Bias (Zeros is usually fine, but small randoms can also help)
        oBias = new Tensor(1, nNeurons)
        oBias.zeros()
        
        # Initialize Gradients
        oGradWeights = new Tensor(nInputSize, nNeurons)
        oGradWeights.zeros()
        oGradBias    = new Tensor(1, nNeurons)
        oGradBias.zeros()
        
    func forward oInputTensor
        oInput = oInputTensor
        
        # Y = X * W
        oOutput = oInput.matmul(oWeights)
        
        # Add Bias (Broadcasting Row-wise)
        biasList = oBias.aData[1]
        for r = 1 to oOutput.nRows
            row = oOutput.aData[r]
            for c = 1 to oOutput.nCols
                row[c] += biasList[c]
            next
        next
        
        return oOutput

    func backward oGradOutput
        # Debugging removed for performance
        
        # 1. Gradients for Weights
        oInputCopy = oInput.copy()
        oInputCopy.transpose()
        oGradWeights = oInputCopy.matmul(oGradOutput)
        
        # 2. Gradients for Bias
        oGradBias = oGradOutput.sum(0) 
        
        # 3. Gradient for Input
        oWeightsCopy = oWeights.copy()
        oWeightsCopy.transpose()
        dInput = oGradOutput.matmul(oWeightsCopy)
        
        return dInput