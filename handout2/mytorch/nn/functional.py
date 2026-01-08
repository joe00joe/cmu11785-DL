import numpy as np

import mytorch.tensor as tensor
from mytorch.autograd_engine import Function

def get_conv1d_output_size(input_size, kernel_size, stride):
    """Gets the size of a Conv1d output.

    Notes:
        - This formula should NOT add to the comp graph.
        - Yes, Conv2d would use a different formula,
        - But no, you don't need to account for Conv2d here.
        
        - If you want, you can modify and use this function in HW2P2.
            - You could add in Conv1d/Conv2d handling, account for padding, dilation, etc.
            - In that case refer to the torch docs for the full formulas.

    Args:
        input_size (int): Size of the input to the layer
        kernel_size (int): Size of the kernel
        stride (int): Stride of the convolution

    Returns:
        int: size of the output as an int (not a Tensor or np.array)
    """
    # TODO: implement the formula in the writeup. One-liner; don't overthink
    return (input_size - kernel_size) // stride + 1
    
    
def im2col_1d(input_data, kernel_size, stride=1, padding=0):
    """
    1D im2col：将输入序列的局部卷积窗口展开为列矩阵
    参数：
        input_data: 输入序列，形状 (N, C_in, L_in)
                    N: 批次，C_in: 输入通道，L_in: 序列长度
        kernel_size: 卷积核长度（int）
        stride: 滑动步长（int）
        padding: 序列左右对称填充长度（int）
    返回：
        col: 展开后的列矩阵，形状 (N * L_out, C_in * kernel_size)
        L_out: 输出序列长度
    """
    N, C_in, L_in = input_data.shape
    K = kernel_size
    s = stride
    p = padding

    # 计算输出序列长度
    L_out = (L_in + 2 * p - K) // s + 1

    # 输入序列填充
    input_padded = np.pad(input_data, ((0, 0), (0, 0), (p, p)), mode='constant')

    # 初始化展开矩阵
    col = np.zeros((N, C_in, K, L_out), dtype=input_data.dtype)

    # 滑动窗口填充
    for k in range(K):
        k_max = k + s * L_out
        col[:, :, k, :] = input_padded[:, :, k:k_max:s]

    # 调整维度并展平
    col = col.transpose(0, 3, 1, 2).reshape(N * L_out, -1)
    return col, L_out



def col2im_1d(col, input_shape, kernel_size, stride=1, padding=0):
    """
    1D col2im：将展开的列矩阵还原为输入序列形状（im2col逆操作）
    参数：
        col: 展开的列矩阵，形状 (N * L_out, C_in * kernel_size)
        input_shape: 原始输入形状 (N, C_in, L_in)
        kernel_size: 卷积核长度
        stride: 滑动步长
        padding: 填充长度
    返回：
        input_restore: 还原后的序列，形状 (N, C_in, L_in + 2*padding)
    """
    N, C_in, L_in = input_shape
    K = kernel_size
    s = stride
    p = padding
    #print('input_shape:', input_shape)
    L_out = (L_in + 2 * p - K) // s + 1
    #print('L_out', L_out)
    # 重塑列矩阵为高维形式
    col = col.reshape(N, L_out, C_in, K).transpose(0, 2, 3, 1)
    

    img = np.zeros((N, C_in, L_in + 2*p + stride - 1))
    
    for k in range(K):
        k_max = k + s*L_out
        img[:, :, k:k_max:s] += col[:, :, k, :]

    return img[:, :, p:L_in + p]

  
    
class Conv1d(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride):
        """The forward/backward of a Conv1d Layer in the comp graph.
        
        Notes:
            - Make sure to implement the vectorized version of the pseudocode
            - See Lec 10 slides # TODO: FINISH LOCATION OF PSEUDOCODE
            - No, you won't need to implement Conv2d for this homework.
        
        Args:
            x (Tensor): (batch_size, in_channel, input_size) input data
            weight (Tensor): (out_channel, in_channel, kernel_size)
            bias (Tensor): (out_channel,)
            stride (int): Stride of the convolution
        
        Returns:
            Tensor: (batch_size, out_channel, output_size) output data
        """
        # For your convenience: ints for each size
        #batch_size, in_channel, input_size = x.shape
        #out_channel, _, kernel_size = weight.shape
        
        # TODO: Save relevant variables for backward pass
        
        # TODO: Get output size by finishing & calling get_conv1d_output_size()
        # output_size = get_conv1d_output_size(None, None, None)

        # TODO: Initialize output with correct size
        # out = np.zeros(())
        
        # TODO: Calculate the Conv1d output.
        # Remember that we're working with np.arrays; no new operations needed.
        
        # TODO: Put output into tensor with correct settings and return 
        
        N, C_in, L_in = x.data.shape
        C_out, _, K = weight.data.shape

        # 1. im2col 展开输入
        col_x, L_out = im2col_1d(x.data, K, stride)  # (N*L_out, C_in*K)

        # 2. 卷积核展平
        col_w = weight.data.reshape(C_out, -1).T  # (C_in*K, C_out)

        # 3. 矩阵乘法实现卷积 + 加偏置
        out_col = np.dot(col_x, col_w) + bias.data  # (N*L_out, C_out)

        # 4. 重塑为标准 1D CNN 输出形状
        out = out_col.reshape(N, L_out, C_out).transpose(0, 2, 1)  # (N, C_out, L_out)

        # 缓存中间变量
        ctx.save_for_backward(x, weight, bias, tensor.Tensor(col_x), tensor.Tensor(col_w), tensor.Tensor(stride))
        requires_grad = x.requires_grad or weight.requires_grad or bias.requires_grad
        return tensor.Tensor(out,requires_grad=requires_grad,
                                    is_leaf=not requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Finish Conv1d backward pass. It's surprisingly similar to the forward pass.
        x, w, b, col_x_t, col_w_t, stride_t = ctx.saved_tensors
        col_x, col_w , stride = col_x_t.data, col_w_t.data, stride_t.data
        N, C_in, L_in = x.data.shape
        C_out, _, K = w.data.shape
        #print("grad_output.shape:", grad_output.data.shape)
        # 1. 调整输出梯度形状，适配矩阵乘法
        dout_reshaped = grad_output.data.transpose(0, 2, 1).reshape(-1, C_out)  # (N*L_out, C_out)
        #print("dout_reshaped.shape:", dout_reshaped.data.shape)
        # 2. 计算偏置梯度 db
        db = np.sum(dout_reshaped, axis=0)  # (C_out,)

        # 3. 计算卷积核梯度 dw
        dw_flat = np.dot(col_x.T, dout_reshaped)  # (C_in*K, C_out)
        dw = dw_flat.T.reshape(C_out, C_in, K)  # (C_out, C_in, K)

        # 4. 计算输入梯度 dx
        dcol_x = np.dot(dout_reshaped, col_w.T)  # (N*L_out, C_in*K)
        #print("dcol_x.shape:", dcol_x.shape)
        dx= col2im_1d(dcol_x, x.data.shape, K, stride)  # (N, C_in, L_in)
      

        return tensor.Tensor(dx), tensor.Tensor(dw), tensor.Tensor(db)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        b_data = np.divide(1.0, np.add(1.0, np.exp(-a.data)))
        ctx.out = b_data[:]
        b = tensor.Tensor(b_data, requires_grad=a.requires_grad)
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        b = ctx.out
        grad = grad_output.data * b * (1-b)
        return tensor.Tensor(grad)
    
class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        b = tensor.Tensor(np.tanh(a.data), requires_grad=a.requires_grad)
        ctx.out = b.data[:]
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.out
        grad = grad_output.data * (1-out**2)
        return tensor.Tensor(grad)


def unbroadcast(grad, shape, to_keep=0):
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        requires_grad = a.requires_grad
        b = tensor.Tensor(a.data.T, requires_grad=requires_grad,
                                    is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.T)

class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                                                 is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.reshape(ctx.shape)), None

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.log(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(grad_output.data / a.data)

"""EXAMPLE: This represents an Op:Add node to the comp graph.

See `Tensor.__add__()` and `autograd_engine.Function.apply()`
to understand how this class is used.

Inherits from:
    Function (autograd_engine.Function)
"""
class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor') or \
            a.data.shape != b.data.shape:
            raise Exception("Both args must be Tensors and have same shape: {}, {}".format(type(a).__name__, type(b).__name__))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data - b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = (-1) * np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only log of tensor is supported")
        ctx.axis = axis
        ctx.shape = a.shape
        if axis is not None:
            ctx.len = a.shape[axis]
        ctx.keepdims = keepdims
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.sum(axis = axis, keepdims = keepdims), \
                          requires_grad=requires_grad, is_leaf=not requires_grad)
        #print(a.shape, c.shape)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        grad_out = grad_output.data

        if (ctx.axis is not None) and (not ctx.keepdims):
            grad_out = np.expand_dims(grad_output.data, axis=ctx.axis)
        else:
            grad_out = grad_output.data.copy()

        grad = np.ones(ctx.shape) * grad_out

        assert grad.shape == ctx.shape
        # Take note that gradient tensors SHOULD NEVER have requires_grad = True.
        return tensor.Tensor(grad), None, None



# TODO: Implement more Functions below
class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor') or \
            a.data.shape != b.data.shape:
            raise Exception("Both args must be Tensors and have same shape: {}, {}".format(type(a).__name__, type(b).__name__))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data * b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = b.data * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = a.data * grad_output.data

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b

class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor') or \
            a.data.shape != b.data.shape:
            raise Exception("Both args must be Tensors and have same shape: {}, {}".format(type(a).__name__, type(b).__name__))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data / b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = grad_output.data / b.data
        # dL/db = dout/db * dL/dout
        grad_b = (-1) * ( grad_output.data / b.data) * (a.data / b.data)

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b
    
class Matmul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor') :
            raise Exception("Both args must be Tensors : {}, {}".format(type(a).__name__, type(b).__name__))
        a_data , b_data= a.data, b.data
        assert a_data.ndim == 2 and b_data.ndim == 2
        assert a_data.shape[1] == b_data.shape[0]
        
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)
        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(np.dot(a.data, b.data), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # calculate gradient of output w.r.t. each input

        grad_a = grad_output.data @ b.data.T
        grad_b = a.data.T @ grad_output.data
        
        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b

class Relu(Function):
    @staticmethod
    def forward(ctx, a):
        # Check that inputs are tensors of same shape
        if not type(a).__name__ == 'Tensor'  :
            raise Exception(" args must be Tensors : {} ".format(type(a).__name__))
        
        
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a)
        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad 
        c = tensor.Tensor(np.maximum(a.data, 0), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c
    
    @staticmethod
    def backward(ctx, grad_output):
        a,= ctx.saved_tensors
        # calculate gradient of output w.r.t. each input
        assert grad_output.data.shape == a.data.shape
        mask = a.data > 0
        grad_a  = grad_output.data * mask.astype(int)
        
        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))

        return grad_a
    

class Relu(Function):
    @staticmethod
    def forward(ctx, a):
        # Check that inputs are tensors of same shape
        if not type(a).__name__ == 'Tensor'  :
            raise Exception(" args must be Tensors : {} ".format(type(a).__name__))
        
        
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a)
        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad 
        c = tensor.Tensor(np.maximum(a.data, 0), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c
    
    @staticmethod
    def backward(ctx, grad_output):
        a,= ctx.saved_tensors
        # calculate gradient of output w.r.t. each input
        assert grad_output.data.shape == a.data.shape
        mask = a.data > 0
        grad_a  = grad_output.data * mask.astype(int)
        
        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))

        return grad_a

def batch_softmax(y_logits):
    """
    批量稳定Softmax实现
    参数：y_logits (N, C) 原始logits
    返回：y_pred (N, C) Softmax预测概率
    """
    logits_max = np.max(y_logits, axis=1, keepdims=True)
    logits_shifted = y_logits - logits_max
    exp_logits = np.exp(logits_shifted)
    exp_logits_sum = np.sum(exp_logits, axis=1, keepdims=True)
    return exp_logits / exp_logits_sum

def batch_cross_entropy_onehot(y_pred, y_true_onehot, epsilon=1e-10):
    """
    批量交叉熵损失（独热标签）
    参数：
        y_pred (N, C) Softmax预测概率
        y_true_onehot (N, C) 独热标签
    返回：
        batch_avg_loss 标量 平均损失
    """
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    sample_losses = -np.sum(y_true_onehot * np.log(y_pred_clipped), axis=1)
    return np.mean(sample_losses)

class XELoss(Function):
    @staticmethod
    def forward(ctx, predicted, target):
        # Check that inputs are tensors of same shape
        if not (type(predicted).__name__ == 'Tensor' and type(target).__name__ == 'Tensor') :
            raise Exception("Both args must be Tensors : {}, {}".format(type(predicted).__name__, type(target).__name__))
        ctx.save_for_backward(predicted, target)

        batch_size, num_classes = predicted.shape
        target_onehot = to_one_hot(target,num_classes )            
        y_pred = batch_softmax(predicted.data)
        batch_loss_split = batch_cross_entropy_onehot(y_pred, target_onehot.data)
        
        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = predicted.requires_grad 
        c = tensor.Tensor(batch_loss_split, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c
    
    @staticmethod
    def backward(ctx, grad_output):
        predicted, target= ctx.saved_tensors
        # calculate gradient of output w.r.t. each input
        batch_size, num_classes = predicted.shape
        target_onehot = to_one_hot(target, num_classes)
        y_pred = batch_softmax(predicted.data)
        grad_p = (y_pred - target_onehot.data) / batch_size
        # the order of gradients returned should match the order of the arguments
        grad_p = tensor.Tensor(unbroadcast(grad_p, predicted.shape))

        return grad_p

def cross_entropy(predicted, target):
    """Calculates Cross Entropy Loss (XELoss) between logits and true labels.
    For MNIST, don't call this function directly; use nn.loss.CrossEntropyLoss instead.

    Args:
        predicted (Tensor): (batch_size, num_classes) logits
        target (Tensor): (batch_size,) true labels

    Returns:
        Tensor: the loss as a float, in a tensor of shape ()
    """
    batch_size, num_classes = predicted.shape

    # Tip: You can implement XELoss all here, without creating a new subclass of Function.
    #      However, if you'd prefer to implement a Function subclass you're free to.
    #      Just be sure that nn.loss.CrossEntropyLoss calls it properly.

    # Tip 2: Remember to divide the loss by batch_size; this is equivalent
    #        to reduction='mean' in PyTorch's nn.CrossEntropyLoss
    

    raise Exception("TODO: Implement XELoss for comp graph")

def to_one_hot(arr, num_classes):
    """(Freebie) Converts a tensor of classes to one-hot, useful in XELoss

    Example:
    >>> to_one_hot(Tensor(np.array([1, 2, 0, 0])), 3)
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [1, 0, 0]]

    Args:
        arr (Tensor): Condensed tensor of label indices
        num_classes (int): Number of possible classes in dataset
                           For instance, MNIST would have `num_classes==10`
    Returns:
        Tensor: one-hot tensor
    """
    arr = arr.data.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1
    return tensor.Tensor(a, requires_grad = True)

