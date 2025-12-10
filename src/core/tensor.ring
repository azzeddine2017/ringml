# File: src/core/tensor.ring
# Description: Core Tensor class - Switching Add/Sub to Manual for Safety
# Author: Code Gear-1

load "fastpro.ring"

class Tensor
    aData   = []    
    nRows   = 0     
    nCols   = 0     

    func init nR, nC
        nRows = nR
        nCols = nC
        aData = list(nRows)
        for i = 1 to nRows
            aData[i] = list(nCols)
        next
        return self
        
    func copy
        oNew = new Tensor(nRows, nCols)
        for r = 1 to nRows
            oNew.aData[r] = aData[r] 
        next
        return oNew

    # --- Initialization ---
    func random
        updateList(aData, :random, :matrix)
        return self

    func zeros
        aData = updateList(aData, :zerolike, :matrix)
        return self

    func fill nVal
        for r = 1 to nRows
            for c = 1 to nCols
                aData[r][c] = nVal
            next
        next
        return self

    # --- Math Operations ---
    
    func add oTensor
        checkDimensions(oTensor)
        # Manual Add for 100% safety
        for r = 1 to nRows
            for c = 1 to nCols
                aData[r][c] += oTensor.aData[r][c]
            next
        next
        return self

    func sub oTensor
        checkDimensions(oTensor)
        # Manual Sub for 100% safety (A - B)
        for r = 1 to nRows
            for c = 1 to nCols
                aData[r][c] -= oTensor.aData[r][c]
            next
        next
        return self

    func mul oTensor
        checkDimensions(oTensor)
        for r = 1 to nRows
             for c = 1 to nCols
                 aData[r][c] *= oTensor.aData[r][c]
             next
        next
        return self

    func scalar_mul nVal
        for r = 1 to nRows
            for c = 1 to nCols
                aData[r][c] *= nVal
            next
        next
        return self

    func sum nAxis
        newData = updateList(aData, :sum, :matrix, nAxis)
        if nAxis = 1 
            oRes = new Tensor(nRows, 1)
        else
            oRes = new Tensor(1, nCols)
        ok
        oRes.aData = newData
        return oRes
        
    func square
        aData = updateList(aData, :square, :matrix)
        return self
        
    func mean
        nRes = updateList(aData, :mean, :matrix)
        return nRes

    # --- Matrix Operations ---
    func matmul oTensor
        if nCols != oTensor.nRows
            raise("Dimension Mismatch in MatMul: Cols " + nCols + " != Rows " + oTensor.nRows)
        ok
        newData = updateList(aData, :mul, :matrix, oTensor.aData)
        oRes = new Tensor(nRows, oTensor.nCols)
        oRes.aData = newData
        return oRes

    func transpose
        nNewRows = nCols
        nNewCols = nRows
        aNewData = list(nNewRows)
        for i = 1 to nNewRows
            aNewData[i] = list(nNewCols)
        next
        for r = 1 to nRows
            for c = 1 to nCols
                aNewData[c][r] = aData[r][c]
            next
        next
        aData = aNewData
        nRows = nNewRows
        nCols = nNewCols
        return self

    # --- Activation Derivatives ---
    func sigmoid
        aData = updateList(aData, :sigmoid, :matrix)
        return self

    func sigmoid_prime
        aData = updateList(aData, :sigmoidprime, :matrix)
        return self

    func relu
        aData = updateList(aData, :relu, :matrix)
        return self
        
    func relu_prime
        aData = updateList(aData, :reluprime, :matrix)
        return self
        
    func tanh
        aData = updateList(aData, :tanh, :matrix)
        return self

    # --- Utilities ---
    func checkDimensions oTensor
        if nRows != oTensor.nRows or nCols != oTensor.nCols
            raise("Dimension Mismatch: (" + nRows + "," + nCols + ") vs (" + oTensor.nRows + "," + oTensor.nCols + ")")
        ok

    func print
        see "Tensor Shape: (" + nRows + ", " + nCols + ")" + nl
        if nRows <= 10 and nCols <= 10
            see aData
        else
            see "Data is too large to display fully." + nl
        ok
        see nl