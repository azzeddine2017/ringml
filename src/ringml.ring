# File: src/ringml.ring
# Description: Main entry point for RingML library

load "fastpro.ring"
load "core/tensor.ring"
load "layers/layer.ring"
load "layers/dense.ring"
load "layers/activation.ring"
load "model/sequential.ring"
load "loss/mse.ring"
load "optim/sgd.ring"

func RingML_Info
    see "RingML Library v1.0 - Ready for Training" + nl