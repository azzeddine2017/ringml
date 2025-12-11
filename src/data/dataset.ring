# File: src/data/dataset.ring
# Description: Dataset and DataLoader for batch processing
# Author: Code Gear-1


class Dataset
    # Base class for datasets
    func len 
        raise("Method len() not implemented")
    func getData itemIndex
        raise("Method getData() not implemented")

class TensorDataset from Dataset
    oInputs
    oTargets
    nSamples

    func init oInTensor, oTargetTensor
        oInputs  = oInTensor
        oTargets = oTargetTensor
        nSamples = oInputs.nRows
        
    func len
        return nSamples
        
    func getData nIdx
        # Returns a list [InputRow, TargetRow]
        # We need to extract specific row as a new Tensor (1, Cols)
        
        # Extract Input Row
        oInRow = new Tensor(1, oInputs.nCols)
        oInRow.aData[1] = oInputs.aData[nIdx] # Copy reference of the list
        
        # Extract Target Row
        oTargetRow = new Tensor(1, oTargets.nCols)
        oTargetRow.aData[1] = oTargets.aData[nIdx]
        
        return [oInRow, oTargetRow]

class DataLoader
    oDataset
    nBatchSize
    nBatches
    
    func init oData, nBatch
        oDataset   = oData
        nBatchSize = nBatch
        nTotal     = oDataset.len()
        # Calculate number of batches (Ceiling)
        nBatches   = ceil(nTotal / nBatchSize)

    func getBatch nBatchIndex
        # nBatchIndex starts from 1
        nStart = (nBatchIndex - 1) * nBatchSize + 1
        nEnd   = nStart + nBatchSize - 1
        
        if nEnd > oDataset.len()
            nEnd = oDataset.len()
        ok
        
        nCurrentBatchSize = nEnd - nStart + 1
        
        # Create Batch Tensors
        # We assume dataset returns [InputTensor, TargetTensor]
        # But for performance, we slice the original big tensor data directly
        
        # 1. Prepare Input Batch Tensor
        oFirstItem = oDataset.getData(1) # Just to check dims
        nFeats     = oFirstItem[1].nCols
        nLabels    = oFirstItem[2].nCols
        
        oBatchInputs = new Tensor(nCurrentBatchSize, nFeats)
        oBatchTargets = new Tensor(nCurrentBatchSize, nLabels)
        
        # Copy data
        nCurrentRow = 1
        for i = nStart to nEnd
            # Direct access to dataset internal tensors is faster if we cast it
            # But sticking to generic API:
            item = oDataset.getData(i)
            oBatchInputs.aData[nCurrentRow]  = item[1].aData[1]
            oBatchTargets.aData[nCurrentRow] = item[2].aData[1]
            nCurrentRow++
        next
        
        return [oBatchInputs, oBatchTargets]