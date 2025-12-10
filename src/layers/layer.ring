# File: src/layers/layer.ring
# Description: Base abstract class for all neural network layers
# Author: Code Gear-1

class Layer
    # Every layer must implement forward pass
    func forward oInputTensor
        raise("Error: forward() method must be implemented by the child class")
        
    # Every layer must implement backward pass
    func backward oGradientTensor
        raise("Error: backward() method must be implemented by the child class")