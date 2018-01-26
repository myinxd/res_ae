# ResAE--Residual Auto Encoder
This repo aims to realize the auto encoder structure with Residual Network (ResNet) by the [TensorFlow](https://www.tensorflow.org) library. The residual loss strategy in the ResNet definately benefits the training for very deep neural networks and keep it from the saturation and performance decreasing problems.

## Which techs will be embedded in the network?
Some tricks will be applied for better training the network to avoide over-fitting, accelerate convergence to local optima, etc. 
- [ReLU activation function](https://en.wikipedia.org/wiki/Rectifier_\(neural_networks\))
- [Batch normalization](http://blog.mazhixian.me/2018/01/23/batch-normalization-with-tensorflow/)
- [Exponential decreacing learning rate](http://blog.mazhixian.me/2018/01/19/adjustable-learning-rate-for-deep-learning-by-tensorflow/)
- [Dropout](https://en.wikipedia.org/wiki/Dropout_\(neural_networks\))
- [Regularization](https://en.wikipedia.org/wiki/Regularization_\(mathematics\))

## Construction of the ResAE
Firstly the bottleneck structure, the basis of the ResNet, is realized. Then the block class is built, composed of multiple bottlenecks. By means of them, the ResNet based encoding part can be constructed. As for the decoder part, which is conventionally as the symmetry as the encoder, it can be formed by reversing the encoder. A diagram is illustrated as follow, which is similar to the famous [skip connection](https://arxiv.org/abs/1606.08921)

<center>
<img src="https://github.com/myinxd/res_ae/blob/master/images/fig_diagram.png?raw=true" height=100 width=720>
</center>


## Packages to be used
Some python packages should be installed before appying the nets, which are listed as follows,
- [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/)
- [matplotlib](http://www.matplotlib.org)
- [Tensorflow](http://www.tensorflow.org)

Also, [CUDA](http://develop.nvidia.org/cuda) is required if you want to run the codes by GPU, a Chinese [guide](http://www.mazhixian.me/2017/12/13/Install-tensorflow-with-gpu-library-CUDA-on-Ubuntu-16-04-x64/) is provided here..

## Usage
<TODO>

## Author
- Zhixian MA <`zx at mazhixian.me`>

## License
Unless otherwise declared:

- Codes developed are distributed under the [MIT license](https://opensource.org/licenses/mit-license.php);
- Documentations and products generated are distributed under the [Creative Commons Attribution 3.0 license](https://creativecommons.org/licenses/by/3.0/us/deed.en_US);
- Third-party codes and products used are distributed under their own licenses.
