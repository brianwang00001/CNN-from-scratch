# Convolutional Neural Network (CNN) from scratch

<br> This is a fully NumPy-based CNN model with PyTorch-style APIs. I tried to replicate some functions in the torch library like nn.Sequential, nn.Conv2d and a naive version of Autograd engine. You might find some of the expressions in this programme quite familiar. Btw my favourite one is loss.backward() :)

I used this library to build a LeNet, and trained with MNIST datasets. 
<br> ---------------------------
<br> CNN architechture: LeNet 
<br> Datasets: MNIST 
<br> Training duration: 1 epoch 
<br> Test accuracy: ~97% 
<br> ---------------------------
<br> Training this model for 1 epoch takes roughly 7 minutes on my Apple M1 chip, while training an equivalent model using PyTorch takes 20 seconds. 

(Note that I replaced the sigmoid in the original LeNet with ReLU, and used MaxPooling rather than AveragePooling. Other model specs are completely the same.)
