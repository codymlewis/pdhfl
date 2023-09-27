# Model Specifications

Dense Neural Network Structure with Partitioning Specifications. Clients locally train their partition using SGD with a learning rate of $0.1$ and a batch size of 128.

| **Layer type** | **Hyperparameters** |
|----------------|---------------------|
| Hidden Layer (Dense) | $[neurons = p_w \cdot 1000, activation = \text{ReLU}] \times p_d \cdot 10$ |
| Output Layer (Dense) | $[neurons = \text{No. Classes}, activation = \text{Softmax}]$ |

ResNet-RS Structure with Partitioning Specifications. Block groups are connected as specified by the bottleneck block in~\cite{bello2021revisiting,he2016deep}. The squeeze-and-excitation blocks are implemented as specified in~\cite{bello2021revisiting}. The global model parameters are initialized with the ImageNet weights provided by the Keras applications in the ResNet-RS implementation in the Tensorflow Version 2.9.0~\cite{chollet2015keras,tensorflow2015-whitepaper}. %After every convolutional layer, there is a batch normalization, and then the ReLU layer, except when the layer precedes a SE block in which case there is only a batch normalization layer before the SE, then ReLU after the SE. For the convolutional layers, we first specify the kernel shape, then the number of filters, $f$, and lastly the size of the strides, $s$, taken. Clients locally train their partition using a learning rate of $0.01$ and momentum of $0.9$ and a batch size of 128.

| **Layer type** | **Hyperparameters** |
|----------------|---------------------|
| STEM | $[\text{Conv } 3\times3, f = 32, s = 2; \text{Conv } 3\times3, f = 32, s = 1; \text{Conv } 3\times3, f = 64, s = 1; \text{Conv } 3\times3, f = 64, s = 2]$ |
| Block group 0 | $[\text{Conv } 1\times1, f = p_w \cdot 64, s = 1; \text{Conv } 3\times3, f = p_w \cdot 64, s = 1; \text{Conv } 1\times1, f = p_w \cdot 64 \cdot 4, s = 1; \text{SE}, f = 64, ratio = 0.25] \times 3$ |
| Block group 1 | $[\text{Conv } 1\times1, f = p_w \cdot 128, s = 1; \text{Conv } 3\times3, f = p_w \cdot 128, s = 2; \text{Conv } 1\times1, f = p_w \cdot 128 \cdot 4, s = 1; \text{SE}, f = 128, ratio = 0.25] \times 4$ |
| Block group 2 | $[\text{Conv } 1\times1, f = p_w \cdot 256, s = 1; \text{Conv } 3\times3, f = p_w \cdot 256, s = 2; \text{Conv } 1\times1, f = p_w \cdot 256 \cdot 4, s = 1; \text{SE}, f = 256, ratio = 0.25] \times p_d \cdot 6$ |
| Block group 3 | $[\text{Conv } 1\times1, f = p_w \cdot 512, s = 1; \text{Conv } 3\times3, f = p_w \cdot 512, s = 2; \text{Conv } 1\times1, f = p_w \cdot 512 \cdot 4, s = 1; \text{SE}, f = 512, ratio = 0.25] \times 3$ |
| Output | $[\text{Global Average Pooling}; \text{Dropout, }rate = 0.25; \text{Dense, }neurons = \text{No. Classes}; \text{Softmax}]$ |
