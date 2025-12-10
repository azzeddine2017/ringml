# File: src/model/sequential.ring
# Description: Sequential Model Container
# Author: Code Gear-1

class Sequential
    aLayers = []

    func add oLayer
        aLayers + oLayer
        return self

    func forward oInput
        oCurrent = oInput
        for oLayer in aLayers
            oCurrent = oLayer.forward(oCurrent)
        next
        return oCurrent

    func backward oGradOutput
        oCurrentGrad = oGradOutput
        # Iterate backwards: from last layer to first
        for i = len(aLayers) to 1 step -1
            oCurrentGrad = aLayers[i].backward(oCurrentGrad)
        next
        return oCurrentGrad
        
    func getLayers
        return aLayers