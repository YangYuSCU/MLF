Python Implementation of the paper "Multilevel Neural Corrections with Adaptive Sampling for Partial Differential Equations".

1. the necessary environment:
pytorch = 2.4.0,
cuda = 12.8 
scipy = 1.15.1 ,
matplotlib = 3.10.0,
numpy = 1.26.3 .

2. For installation instructions regarding the SOAP optimization method, please refer to "Soap: Improving and stabilizing shampoo using Adam" or install `pytorch_optimizer` directly.
For installation instructions regarding the SSBroyden optimization method, please refer to "Unveiling the optimization process of physics-informed neural networks: How accurate and competitive can PINNs be?"

3.After setting up the necessary environment, run `/Poisson/tanhPINN/train.py`.
