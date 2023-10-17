# Model Specifications

Dense Neural Network Structure with Partitioning Specifications. Clients locally train their partition using SGD with a learning rate of $0.1$ and a batch size of 128.

## FCN

| **Layer type** | **Hyperparameters** |
|----------------|---------------------|
| Hidden Layer (Dense) | $[neurons = p_w \cdot 1000, activation = \text{ReLU}] \times p_d \cdot 10$ |
| Output Layer (Dense) | $[neurons = \text{No. Classes}, activation = \text{Softmax}]$ |


## CNN

The VGG based network

| **Layer type** | **Hyperparameters** |
|----------------|---------------------|
| Conv block | $[Conv(c=32 \times 2^l \times p_w, k=3 \times 3), ReLU, Conv(c=32 \times 2^l \times p_w, k=3 \times 3), ReLU, MaxPool(k=2 \times 2, s=2 \times 2)]_{l = 1}^{5 \times p_d}$ |
| Ordered Padding | |
| Dense | neurons=128, activation=ReLU |
| Dense | neurons=128, activation=ReLU |
| Output Layer (Dense) | [neurons = No. Classes, activation=Softmax] |


## DenseNet

| **Layer type** | **Hyperparameters** |
|----------------|---------------------|
| Zero Padding | |
| Convolutional | $c=64, k=7 \times 7, s=2 \times 2$ |
| Group normalization | groups=32 |
| ReLU | |
| Zero Padding | |
| Max Pooling | $k=3 \times 3, s=2 \times 2$ |
| Dense Block | $blocks=6 \times p_d, \text{growth rate}=32 \times p_w$ |
| Transitional Block | $reduction=0.54 |
| Dense Block | $blocks=12 \times p_d, \text{growth rate}=32 \times p_w$ |
| Transitional Block | $reduction=0.5$ |
| Dense Block | $blocks=24 \times p_d, \text{growth rate}=32 \times p_w$ |
| Transitional Block | $reduction=0.5$ |
| Dense Block | $blocks=16 \times p_d, \text{growth rate}=32 \times p_w$ |
| Transitional Block | $reduction=0.5$ |
| Group normalization | $groups=32$ |
| ReLU | |
| Global Average Pooling | |
| Output Layer (Dense) | $[neurons = \text{No. Classes}, activation = \text{Softmax}]$ |
