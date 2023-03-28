"""
A solely NumPy-based, PyTorch-like library for building CNN
"""
import numpy as np
import time

# in most part of this lib, 
#   indata = input data
#   outdata = output data
#   ingrad = local gradient of input data,
#   outgrad = local gradient of output data

# --------------------------------------------------------------------------------------------------
# input size: (N, Cin, H, W) = (batch size, input channels, height, width)
# output size: (N, Cout, Hout, Wout) = (batch size, output channels, output height, output width)
class Conv2d:

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, model_mode='forward'):
        # specs
        self.stride = stride
        self.padding = padding 
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        # params
        self.kernel = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.bias = np.zeros((out_channels)) if bias else None
        self.kernel_grad = np.zeros_like(self.kernel)
        self.bias_grad = np.zeros_like(self.bias) if bias else None
        # model mode
        self.model_mode = model_mode

        # layer information
        self.layer_info = f'Conv2d({in_channels}, {out_channels}, kernel_size=({kernel_size}, {kernel_size}), stride=({stride}, {stride}), padding=({padding, padding}))'
        self.name = 'Conv2d    '
 
    def __repr__(self):
        return self.layer_info

    def __call__(self, indata):
        assert (self.model_mode == 'forward' or self.model_mode == 'backward')
        if self.model_mode == 'forward':
            self.indata = indata
            indata_padded = self.zero_padding(self.indata, self.padding)
            self.outdata = self.convolve(indata_padded, self.kernel, self.stride) + self.bias.reshape(1, self.out_channels, 1, 1)
            return self.outdata
        elif self.model_mode == 'backward':
            # now indata becomes the gradient for the output data, aka. outgrad
            self.kernel_grad, ingrad = self.conv_backprop(indata)
            self.bias_grad = np.sum(indata, axis=(0, 2, 3))
            return ingrad
        
    def parameters(self):
        return [self.kernel] + ([] if self.bias is None else [self.bias])
    
    def gradients(self):
        return [self.kernel_grad] + ([] if self.bias is None else [self.bias_grad])
    
    def convolve(self, indata, kernel, stride=1):
        N, in_channels, H, W = indata.shape
        ksize = kernel.shape[-1]
        out_channels = kernel.shape[0]
        S = stride
        # output size
        Hout = int((H - ksize) / S + 1)
        Wout = int((W - ksize) / S + 1)
        assert (Hout == (H - ksize) / S + 1 and Wout == (W - ksize) / S + 1)

        # transform multidimentional inputs into two 2d matrices
        x_col = np.zeros((ksize*ksize*in_channels, Hout*Wout*N))
        w_row = kernel.reshape(out_channels, ksize*ksize*in_channels)

        # assign values to w_row
        for i in range(N):
            for j in range(Hout):
                for k in range(Wout):
                    x_col[:, i*(Hout*Wout)+j*Wout+k] = indata[i, :, j*S:j*S+ksize, k*S:k*S+ksize].ravel()
        
        # do the necessary computation at one time
        outdata = w_row @ x_col
        
        # Note that if we just use outdata.reshape(N, out_channels, Hout, Wout), the dimensional order will 
        # have problem because of the reshaping mechanism. Hence, we use reshape(out_channels, N, ", ") instead
        # then transpose the first and second axis
        outdata = outdata.reshape(out_channels, N, Hout, Wout)
        outdata = np.transpose(outdata, (1, 0, 2, 3))
        return outdata
    
    def conv_backprop(self, outgrad):
        S = self.stride
        P = self.padding
        
        # insert S zero rows and columns in outgrad 
        Hout_new, Wout_new = [i+(i-1)*(S-1) for i in outgrad.shape[-2:]]
        outgrad_new = np.zeros((outgrad.shape[0], outgrad.shape[1], Hout_new, Wout_new))
        outgrad_new[:, :, ::S, ::S] = outgrad
        # compute kernel grad
        indata_padded = self.zero_padding(self.indata, self.padding)
        kernel_grad = self.convolve(np.transpose(indata_padded, (1, 0, 2, 3)), np.transpose(outgrad_new, (1, 0, 2, 3)))
        kernel_grad = np.transpose(kernel_grad, (1, 0, 2, 3))
        

        # zero pad outgrad_new
        outgrad_new_padded = self.zero_padding(outgrad_new, self.kernel_size-1)
        # flip kernel horizontally and vertically
        kernel_flipped = np.flip(np.flip(self.kernel, 2), 3)
        ingrad = self.convolve(outgrad_new_padded, np.transpose(kernel_flipped, (1, 0, 2, 3)))

        # delete the padded part of ingrad
        ingrad = self.un_padding(ingrad)

        return kernel_grad, ingrad

    def zero_padding(self, indata, P):
        N, in_channels, H, W = indata.shape
        padded_indata = np.zeros((N, in_channels, H+2*P, W+2*P))
        padded_indata[:, :, P:P+H, P:P+W] = indata
        return padded_indata
    
    def un_padding(self, indata):
        if self.padding != 0:
            out = indata[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            out = indata
        return out 
    
# --------------------------------------------------------------------------------------------------
class Linear:

    def __init__(self, fan_in, fan_out, bias=True, model_mode='forward'):
        self.weight = np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)
        self.bias = np.zeros((1, fan_out)) if bias else None
        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias) if bias else None
        self.model_mode = model_mode
        self.layer_info = f'Linear(in_features={fan_in}, out_features={fan_out}, bias={bias})'
        self.name = 'Linear    '

    def __repr__(self):
        return self.layer_info

    def __call__(self, x):
        assert (self.model_mode == 'forward' or self.model_mode == 'backward')
        if self.model_mode == 'forward':
            # forward mode
            # x will be data for input
            self.input = x 
            self.output = self.input @ self.weight
            if self.bias is not None:
                self.output += self.bias
            return self.output
        elif self.model_mode == 'backward':
            # backprop mode
            # x will be gradient of output 
            if self.bias is not None:
                self.bias_grad = np.mean(x, axis=0)
            self.weight_grad = self.input.T @ x
            return x @ self.weight.T

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
    def gradients(self):
        return [self.weight_grad] + ([] if self.bias is None else [self.bias_grad])
     
# --------------------------------------------------------------------------------------------------
# input size: (N, C, H, W) = (batch size, channels, height, width)
# output size: (N, C, Hout, Wout) = (batch size, channels, output height, output width)
class MaxPool2d:

    def __init__(self, kernel_size=2, stride=2, padding=0, model_mode='forward'):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.model_mode = model_mode
        self.layer_info = f'MaxPool2d(kernel_size={kernel_size}, stride={stride}, padding={padding})'
        self.name = 'MaxPool2d '

    def __repr(self):
        return self.layer_info

    def __call__(self, indata):
        assert (self.model_mode == 'forward' or self.model_mode == 'backward')
        if self.model_mode == 'forward':
            self.indata = indata
            outdata = self.maxpool(self.indata)
            return outdata
        elif self.model_mode == 'backward':
            outgrad = indata
            ingrad = self.maxpool_backprop(outgrad)
            return ingrad
    
    # maxpool, with zero padding process
    def maxpool(self, indata):
        # zero padding
        padded_indata = self.zero_padding(indata, self.padding)

        ksize = self.kernel_size
        S = self.stride
        N, in_channels, H, W = padded_indata.shape

        # calculate outdata shape
        Hout, Wout = int((H-ksize)/S+1), int((W-ksize)/S+1)
        assert (Hout == (H-ksize)/S+1 and Wout == (W-ksize)/S+1)
        outdata = np.zeros((N, in_channels, Hout, Wout))
        for i in range(Hout):
            for j in range(Wout):
                outdata[:, :, i, j] = padded_indata[:, :, i*S:i*S+ksize, j*S:j*S+ksize].max(axis=(2,3))
        return outdata

    # maxpool backprop dev.
    def maxpool_backprop(self, outgrad):
        # zero padding
        padded_indata = self.zero_padding(self.indata, self.padding)

        ksize = self.kernel_size
        S = self.stride
        N, in_channels, H, W = padded_indata.shape
        Hout, Wout = outgrad.shape[-2:]

        # input gradient
        ingrad = np.zeros_like(padded_indata)
        for i in range(Hout):
            for j in range(Wout):
                # part of indata
                indata_chunk = padded_indata[:, :, i*S:i*S+ksize, j*S:j*S+ksize]

                # max index for this part of data. dim = (N * in_channels)
                # index number = 0,1,...,(ksize^2-1)
                max_idx = indata_chunk.reshape(N*in_channels, ksize*ksize).argmax(1).reshape(N, in_channels)

                # gradient filter: for the chosen max elements allowed tp propagate the gradient, grad_filter = 1 
                #                  otherwise, grad_filter = 0
                grad_filter = np.zeros_like(indata_chunk)

                # convert the max_index to the index for matrix operation
                idx_1, idx_2 = np.unravel_index(max_idx.ravel(), (ksize, ksize))

                # and the index for batch and channel dimension for assigning gradients  
                idx_n, idx_c = np.meshgrid(np.arange(N), np.arange(in_channels))
                idx_n, idx_c = idx_n.T.ravel(), idx_c.T.ravel()
                grad_filter[idx_n, idx_c, idx_1, idx_2] = 1

                # assign output gradient to input
                ingrad[:, :, i*S:i*S+ksize, j*S:j*S+ksize] += grad_filter * outgrad[:, :, i:i+1, j:j+1]

        # remove the padded part of ingrad
        ingrad = self.un_padding(ingrad)

        return ingrad
    
    # add P zero padding to the input data
    def zero_padding(self, indata, P):
        N, in_channels, H, W = indata.shape
        padded_indata = np.zeros((N, in_channels, H+2*P, W+2*P))
        padded_indata[:, :, P:P+H, P:P+W] = indata
        return padded_indata
    
    # remove the zero paddings
    def un_padding(self, indata):
        if self.padding != 0:
            out = indata[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            out = indata
        return out 
    
    def parameters(self):
        return []
    
    def gradients(self):
        return []
    
# --------------------------------------------------------------------------------------------------
# input size = output size
class ReLU:

    def __init__(self, model_mode='forward'):
        self.model_mode = model_mode
        self.layer_info = 'ReLU()'
        self.name = 'ReLU      '

    def __repr(self):
        return self.layer_info

    def __call__(self, indata):
        assert (self.model_mode == 'forward' or self.model_mode == 'backward')
        if self.model_mode == 'forward':
            # remember which input component to propagate gradient
            self.indata_idx = (indata >= 0)
            outdata = np.maximum(0, indata)
            return outdata
        if self.model_mode == 'backward':
            # now indata becomes output gradients
            outgrad = indata 
            ingrad = np.zeros(self.indata_idx.shape)
            ingrad[self.indata_idx] = outgrad[self.indata_idx]
            return ingrad
        
    def parameters(self):
        return []
    
    def gradients(self):
        return []

# --------------------------------------------------------------------------------------------------
# input size = output size
class Flatten:

    def __init__(self, model_mode='forward'):
        self.model_mode = model_mode
        self.layer_info = f'Flatten(start_dim=1, end_dim=-1)'
        self.name = 'Flatten   '

    def __repr(self):
        return self.layer_info

    def __call__(self, indata):
        assert (self.model_mode == 'forward' or self.model_mode == 'backward')
        if self.model_mode == 'forward':
            self.data_shape = indata.shape
            batch_dim = self.data_shape[0]
            flattened_dim = np.product(self.data_shape[1:])
            outdata = indata.reshape(batch_dim, flattened_dim)
            return outdata
        if self.model_mode == 'backward':
            ingrad = indata.reshape(self.data_shape)
            return ingrad
        
    def parameters(self):
        return []
    
    def gradients(self):
        return []

# --------------------------------------------------------------------------------------------------
class Sequential:

    def __init__(self, *arg):
        self.layers = arg
        self.model_mode = 'forward'

        # record of time consumed by forward and backward pass in each layer
        self.forward_time = {f'layer {i}':0 for i,_ in enumerate(self.layers)}
        self.backward_time = {f'layer {i}':0 for i,_ in enumerate(self.layers)}

    def __repr__(self):
        layer_info = 'Sequential( \n'
        for layer in self.layers:
            layer_info += '  '
            layer_info += layer.layer_info
            layer_info += '\n'
        layer_info += ')'
        layer_info +='\n'
        # number of total parameters
        param_count = sum(np.product(param.shape) for param in self.parameters())
        layer_info += f'Number of parameters : {param_count}'
        return layer_info

    def __call__(self, x):
        assert (self.model_mode == 'forward' or self.model_mode == 'backward')
        if self.model_mode == 'forward':
            self.input = x 
            for i, layer in enumerate(self.layers):
                starting_time = time.time()
                x = layer(x)
                self.forward_time[f'layer {i}'] += time.time() - starting_time
            self.out = x
            return self.out 
        elif self.model_mode == 'backward':
            for i, layer in enumerate(reversed(self.layers)):
                starting_time = time.time()
                x = layer(x)
                self.backward_time[f'layer {len(self.layers)-1-i}'] += time.time() - starting_time
            return x 
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def gradients(self):
        return [g for layer in self.layers for g in layer.gradients()]
    
    # switch model between forward and backward mode
    def set_mode(self, mode):
        assert (mode == 'forward' or mode == 'backward')
        self.model_mode = mode
        for layer in self.layers:
            layer.model_mode = self.model_mode

    # save trained model
    def save_model(self, model_name):
        params = self.parameters()
        np.save(model_name+'.npy', np.array(params, dtype=object), allow_pickle=True)

    # load trained model 
    def load_model(self, file_name):
        # load model parameter
        trained_params = np.load(file_name, allow_pickle=True).tolist()
        # number of parameter to overwrite
        num_of_param = len(trained_params)
        for param, trained_param in zip(self.parameters(), trained_params):
            param *= 0
            param += trained_param
    """
    I must point out one bizarre issue in this section. In load_model(), if I use 
    param = trained_param, the 'param' term will not be altered. In fact, if I 
    use the '=' operand, the value cannot be assigned to parameter and it will remain 
    unchanged. Only oprands like *= and += can change its value. If anyone sees this 
    and knows the reason please let me know. Big thanks! 
    """
# --------------------------------------------------------------------------------------------------
class Cross_entropy:

    # functionality 1: calculate the loss
    # functionality 2: loss.backward()!
    def __init__(self, model, indata, label):
        self.model = model
        self.label = label
        model.set_mode('forward')
        self.loss = self.cross_entropy(model(indata), label)

    def __repr__(self):
        return f'{self.loss}'
        
    def cross_entropy(self, indata, label):
        # (indata=logits) -> counts -> probs -> loss
        N = indata.shape[0] # number of training example
        indata -= np.max(indata) # for numerical stability
        counts = np.exp(indata)
        probs = counts / np.sum(counts, axis=1, keepdims=True)
        self.probs = probs # save for later gradient descent
        loss = -np.mean(np.log(probs[np.arange(N), label]))
        return loss

    def backward(self):
        self.model.set_mode('backward') # set model to backward mode 
        N = self.probs.shape[0] # number of training example
        # d(loss)/d(probs)
        probs_grad = self.probs 
        probs_grad[np.arange(N), self.label] -= 1
        # backprop!
        self.model(probs_grad)
        self.model.set_mode('forward') # set model back to forward mode 

    # return loss value
    def item(self):
        return self.loss

# --------------------------------------------------------------------------------------------------
class SGD:

    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def step(self):
        for param, grad in zip(self.model.parameters(), self.model.gradients()):
            param -= self.lr * grad

# --------------------------------------------------------------------------------------------------
# take the datasets as input, partition it into mini batches (shuffle it if shuufle==true).
def DataLoader(images, labels, batch_size, shuffle):
    # total number and dimension of image data
    N, H, W = images.shape
    # order of fetching data
    idx = np.random.choice(np.arange(N), N, replace=False) if shuffle else np.arange(N)
    out = [[images[idx[i * batch_size : (i + 1) * batch_size]].reshape(batch_size, 1, H, W), 
                   labels[idx[i * batch_size : (i + 1) * batch_size]]] for i in range(int(N / batch_size))]   
    return out

