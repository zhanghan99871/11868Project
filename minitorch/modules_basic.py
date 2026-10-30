"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np

from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy, LayerNorm)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor

from typing import Any, Dict, Optional, Sequence, Tuple


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weights : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        ### BEGIN ASSIGN3_2
        # raise NotImplementedError
        weights_init = tensor_from_numpy(np.random.normal(0.0, 1.0, size=(num_embeddings, embedding_dim)), backend=backend)
        self.weights = Parameter(weights_init) 
        ### END ASSIGN3_2
    
    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        ### BEGIN ASSIGN3_2
        # raise NotImplementedError
        x_one_hot = one_hot(x, self.num_embeddings)
        x_one_hot = x_one_hot.view(bs * seq_len, self.num_embeddings)
        output = x_one_hot @ self.weights.value
        return output.view(bs, seq_len, self.embedding_dim)
        ### END ASSIGN3_2

    
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)

        Note: If p_dropout is 0, directly return the input tensor. Otherwise, the random seed may cause problems
        """
        ### BEGIN ASSIGN3_2
        if not self.training or self.p_dropout == 0:
            return x 
        else:
            keep_prob = 1 - self.p_dropout
            mask = tensor_from_numpy(np.random.binomial(n=1, p=keep_prob, size=x.shape), backend=x.backend) 
            return x * mask / keep_prob
        # raise NotImplementedError
        ### END ASSIGN3_2


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weights - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
        """
        def uniform(shape, value):
            r = rand(shape, backend=backend)
            r = r - 0.5 
            return Parameter(r * value * 2)
        self.out_size = out_size
        ### BEGIN ASSIGN3_2
        self.weights = uniform((in_size, out_size), tensor_from_numpy(1 / np.sqrt(in_size), backend=backend))
        self.bias = uniform((out_size, ), tensor_from_numpy(1 / np.sqrt(in_size), backend=backend)) if bias else None
        # raise NotImplementedError
        ### END ASSIGN3_2

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size)
        """
        batch, in_size = x.shape
        ### BEGIN ASSIGN3_2
        x = x.view(batch, in_size) 
        weights = self.weights.value.view(in_size, self.out_size) 
        out = x @ weights
        out = out.view(batch, self.out_size) 
        if self.bias is not None:
            out = out + self.bias.value
        return out 
        # raise NotImplementedError
        ### END ASSIGN3_2


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend, use_fused_kernel=False):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        ### BEGIN ASSIGN3_2
        weight_init = ones((self.dim, ), backend=backend) 
        bias_init = zeros((self.dim, ), backend=backend) 
        self.weights = Parameter(weight_init) 
        self.bias = Parameter(bias_init) 
        self.use_fused_kernel = use_fused_kernel
        # raise NotImplementedError
        ### END ASSIGN3_2

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        batch, dim = x.shape
        ### BEGIN ASSIGN3_2
        if not self.use_fused_kernel:
            x_mean = x.mean(dim=1) 
            x_var = x.var(dim=1) 
            x = (x - x_mean) / ((x_var + self.eps) ** 0.5)
            return self.weights.value * x + self.bias.value
        else:
            return LayerNorm.apply(x, self.weights.value, self.bias.value) 
        # raise NotImplementedError
        ### END ASSIGN3_2
