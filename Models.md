# Model Specifications

Dense Neural Network Structure with Partitioning Specifications. Clients locally train their partition using SGD with a learning rate of $0.1$ and a batch size of 128.

## FCN

| **Layer type** | **Hyperparameters** |
|----------------|---------------------|
| Flatten | Reshape from $n + 1$ dimensions to $2$ |
| Hidden Layer (Dense) | $[neurons = p_w \cdot 1000, activation = \text{ReLU}] \times 10p_d$ |
| Output Layer (Dense) | $[neurons = \text{No. Classes}, activation = \text{Softmax}]$ |


## CNN

The VGG based network

| **Layer type** | **Hyperparameters** |
|----------------|---------------------|
| Conv block | $[Conv(c=32 \times 2^l \times p_w, k=3 \times 3, act = ReLU), Conv(c=32 \times 2^l \times p_w, k=3 \times 3, act = ReLU), MaxPool(k=2 \times 2, s=2 \times 2)] \times 5p_d$ |
| Flatten | Reshape from $n + 1$ dimensions to $2$ |
| Dense | neurons = $128 p_w$, activation = ReLU |
| Dense | neurons = $128 p_w$, activation=ReLU |
| Output Layer (Dense) | $[neurons = \text{No. Classes}, activation = \text{Softmax}]$ |


## DenseNet

| **Layer type** | **Hyperparameters** |
|----------------|---------------------|
| Convolutional | $c=64, k=7 \times 7, s=2 \times 2$ |
| Layer normalization | $\epsilon = 1.001 \times 10^{-5}$, activation = ReLU |
| Max Pooling | $k=3 \times 3, s=2 \times 2$ |
| Dense Block | $blocks=6 \times p_d, \text{growth rate}=32 \times p_w$ |
| Transitional Block | $reduction=0.5$ |
| Dense Block | $blocks=12 \times p_d, \text{growth rate}=32 \times p_w$ |
| Transitional Block | $reduction=0.5$ |
| Dense Block | $blocks=24 \times p_d, \text{growth rate}=32 \times p_w$ |
| Transitional Block | $reduction=0.5$ |
| Dense Block | $blocks=16 \times p_d, \text{growth rate}=32 \times p_w$ |
| Transitional Block | $reduction=0.5$ |
| Layer normalization | $\epsilon = 1.001 \times 10^{-5}, activation = \text{ReLU}$ |
| Global Average Pooling | - |
| Output Layer (Dense) | $[neurons = \text{No. Classes}, activation = \text{Softmax}]$ |
