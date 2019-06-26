
import torch
import torch.nn as nn

__all__ = ['GraphConvolution', 'GraphConvolutionalNetwork']

def padded_vertex_mean(X, masks, vertex_axis=-2, keepdim=True, epsilon=1e-8):
    """ Calculates a mean along all vertices that were not padded.
    
    Paramters:
    ----------
    X : torch.FloatTensor, shape [batch_size, N, D]
        The tensor to normalize.
    masks : torch.FloatTensor, shape [batch_size, N, N]
        The masks for the vertices.
    vertex_axis : int
        The axis of vertices that will be reduced.
    keepdim : bool
        If the rank of X should be kept.
    epsilon : float
        Epsilon to enhance numerical instabilities.
    
    Returns:
    --------
    X_mean : torch.FloatTensor, shape [batch_size, 1, D]
        The vertex mean of X.
    """
    num_vertcies = torch.sum(torch.max(masks, -1, keepdim=True).values, vertex_axis, keepdim=True)
    mean = torch.sum(X, vertex_axis, keepdim=True) / (num_vertcies + epsilon)
    if not keepdim:
        mean = torch.squeeze(mean, dim=vertex_axis)
    return mean

def padded_softmax(X, masks, axis=-1, keepdim=True, epsilon=1e-8):
    """ Applies a softmax that considers padded vertices. 
    
    Paramters:
    ----------
    X : torch.FloatTensor, shape [batch_size, N, D]
        The matrix to normalize.
    masks : torch.FloatTensor, shape [batch_size, N, N]
        The mask that considers padded vertices.
    axis : int
        The axis along which the softmax should be applied.
    keepdim : bool
        If the dimensions of X should be kept.
    epsilon : float
        Epsilon to prevent zero divisions.
    
    Returns:
    --------
    normalized : torch.FloatTonsor, shape [batch_size, ...]
        The normalized data.
    """
    X = torch.exp(-X) * masks
    normalization = torch.sum(X, axis, keepdim=True)
    return X / (normalization + epsilon)


class GraphConvolution(nn.Module):
    """ Module that implements graph convolutions. """

    def __init__(self, input_dim, output_dim, use_bias=True, activation=True, dropout_rate=None, use_batchnorm=False,
        use_residual=True):
        """ Initializes the graph convolution.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features.
        output_dim : int
            Number of output features.
        use_bias : bool
            If to use a bias for the linear transformation.
        activation : bool
            If an activation function should be applied to the embedding.
        dropout_rate : float or None
            If given, the dropout rate for this layer.
        use_batchnorm : bool
            If True, batchnorm that accounts for padded vertices will be applied.
        use_residual : bool
            If the block should implement an additive skip connection. If True, this will only be implemented
            if input_dim == output_dim.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
        if use_batchnorm:
            self.batchnorm = PaddedBatchnorm(output_dim)
        else:
            self.batchnorm = None
        if activation: 
            self.activation = nn.ReLU()
        else:
            self.activation = None
        if dropout_rate:
            self.dropout = nn.modules.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.use_residual = use_residual and input_dim == output_dim

    def forward(self, X, A, masks):
        """ Forward pass.
        
        Parameters:
        -----------
        X : torch.FloatTensor, shape [batch_size, N, D]
            The embedding matrices.
        A : torch.FloatTensor, shape [batch_size, N, N]
            Adjacency matrices for each event.
        masks : torch.FloatTensor, shape [batch_size, N, N]
            The masks for the adjacency matrix.

        Returns:
        --------
        E : torch.floatTensor, shape [batch_size, N, D']
            The embedding output of this layer.
        """
        E = self.linear(X)
        E = torch.bmm(A, E)
        if self.batchnorm:
            E = self.batchnorm(E, masks)
        if self.activation:
            E = self.activation(E)
        if self.use_residual:
            E = E + X
        if self.dropout:
            E = self.dropout(E)
        return E


class PaddedBatchnorm(nn.Module):
    """ Batchnorm that considers padded vertices. """

    def __init__(self, number_features, momentum=0.1, epsilon=1e-8):
        """ Initializes the batchnorm layer that considers padded vertices.
        
        Parameters:
        -----------
        number_features : int
            The number of features the embedding has.
        momentum : float
            Momentum for running mean and variance.
        epsilon : float
            Small value to improve numerical stabilities when performing divisons.
        """
        super().__init__()
        self.register_buffer('running_mean', torch.zeros(number_features))
        self.register_buffer('running_variance', torch.ones(number_features))
        self.beta = nn.Parameter(torch.FloatTensor(number_features))
        self.gamma = nn.Parameter(torch.FloatTensor(number_features))
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)
        self.epsilon = epsilon
        self.momentum = momentum

    def forward(self, X, masks):
        """ """
        if self.training:
            # Calculate mean over vertices and batches, but consider padded vertices
            batch_mean = torch.mean(padded_vertex_mean(X, masks, vertex_axis=-2, keepdim=True), 0, keepdim=True)
            X_centered = X - batch_mean
            batch_variance = torch.mean(padded_vertex_mean(X_centered ** 2, masks, vertex_axis=-2, keepdim=True), 0, keepdim=True)
            X_scaled = X_centered / (torch.sqrt(batch_variance) + self.epsilon)

            # Update running mean and variance
            with torch.no_grad():
                self.running_mean = self.momentum * batch_mean + (1 - self.momentum) * self.running_mean
                self.running_variance = self.momentum * batch_variance + (1 - self.momentum) * self.running_variance
        else:
            # Use estimates from running mean and running variance
            X_scaled = (X - self.running_mean) / (torch.sqrt(self.running_variance) + self.epsilon)
        return (X_scaled + self.beta) * self.gamma



class GaussianKernel(nn.Module):
    """ Applies a kernel to a distance matrix in order to build an adjacency matrix. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inverse_sigma = nn.Parameter(torch.rand(1) * 0.02 + 0.99)
    
    def forward(self, C, masks):
        """ Forward pass through the kernel. 
        
        Parameters:
        -----------
        C : torch.FloatTensor, shape [batch_size, N, num_coordinates]
            Coordinates for all events.
        mask : torch.FloatTensor, shape [batch_size, N, N]
            Padding mask for each adjacency matrix.

        Returns:
        --------
        A : torch.FloatTensor, shape [batch_size, N, N]
            Adjacency matrices for each event.
        """
        D = pairwise_distances(C)
        A = torch.exp(-self.inverse_sigma * D)
        A = padded_softmax(A, -1)
        return A

def pairwise_distances(C):
    """ Calculates a pairwise distance matrix between rows of C. 
    
    Parameters:
    -----------
    C : torch.FloatTensor, shape [batch_size, N, num_coordinates]
        The coordinates to use for pairwise distance calculation.
    
    Returns:
    --------
    D : torch.FloatTensor, shape [batch_size, N, N]
        Pairwise euclidean distances between all N points.
    """
    batch_size, N, num_coordinates = C.size()

    # Expand to a fourth dimension in order to calculate distances
    expanded = C.unsqueeze(-2).expand(batch_size, N, N, num_coordinates)
    return ((expanded - expanded.transpose(-3, -2))**2).sum(-1)



class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, num_input_features, units_graph_convolutions = [64, 64, 64, 64, 64], units_fully_connected = [1], 
        dropout_rate=0.5, use_batchnorm=True, use_residual=True, **kwargs):
        """ Creates a GCN model. 

        Parameters:
        -----------
        num_input_features : int
            Number of features for the input. 
        units_graph_convolutions : list
            The hidden units for each layer of graph convolution.
        units_fully_connected : list
            The hidden units for each fully connected layer. Pass '1' for logistic regression of pooled node features.
        dropout_rate : float
            Dropout rate.
        use_batchnorm : bool
            If batch normalization should be applied.
        use_residual : bool
            If the block should implement an additive skip connection. If True, this will only be implemented
            if num_input_features == num_output_features for a graph convolution layer.
        """
        super().__init__(**kwargs)
        self.kernel = GaussianKernel()
        self.graph_convolutions = torch.nn.ModuleList()
        self.layers_fully_connected = torch.nn.ModuleList()
        for idx, (D, D_prime) in enumerate(zip([num_input_features] + units_graph_convolutions[:-1], units_graph_convolutions)):
            is_last_layer = idx == len(units_graph_convolutions) - 1
            self.graph_convolutions.append(
                GraphConvolution(
                    D, D_prime, use_bias=True, 
                    activation=not is_last_layer, 
                    dropout_rate=None if is_last_layer else dropout_rate,
                    use_batchnorm=use_batchnorm and not is_last_layer
                    ))

        for idx, (D, D_prime) in enumerate(zip([units_graph_convolutions[-1]] + units_fully_connected[:-1], units_fully_connected)):
            is_last_layer = idx == len(units_fully_connected) - 1
            self.layers_fully_connected.append(nn.Linear(D, D_prime, bias=True))
            if not is_last_layer:
                self.layers_fully_connected.append(nn.ReLU())
            else:
                self.layers_fully_connected.append(nn.Sigmoid())
        
    def forward(self, X, D, masks):
        """ Forward pass.
        
        Parameters:
        -----------
        X : torch.FloatTensor, shape [batch_size, N, D]
            The node features.
        D : torch.FloatTensor, shape [batch_size, N, N]
            Pairwise distances for all nodes.
        masks : torch.FloatTensor, shape [batch_size, N, N]
            Masks for the adjacency / distance matrix.
        """

        # Graph convolutions
        A = self.kernel(D, masks)
        for graph_convolution in self.graph_convolutions:
            X = graph_convolution(X, A, masks)

        # Average pooling
        X = padded_vertex_mean(X, masks, vertex_axis=-2, keepdim=False)

        # Fully connecteds, (usually only 1 layer, i.e. logistic regression)
        for layer in self.layers_fully_connected:
            X = layer(X)
        return X
    
            
class GraphConvolutionalNetworkWithGraphFeatures(GraphConvolutionalNetwork):
    """ Subclass of a GCN that also takes graph features and trains a sub-network based on those. """

    def __init__(self, num_input_features_gcn, num_input_features_graph_mlp, units_graph_convolutions = [64, 64, 64, 64, 64], 
        units_fully_connected = [1], units_graph_mlp = [64, 64, 64],
        units_graph_features=None, dropout_rate=0.5, use_batchnorm=True, use_residual=True, **kwargs):
        """ Creates a GCN model. 

        Parameters:
        -----------
        num_input_features_gcn : int
            Number of features for the input of the GCN part.
        num_input_features_graph_mlp : int
            Number of features for the input of the graph MLP part. 
        units_graph_convolutions : list
            The hidden units for each layer of graph convolution.
        units_fully_connected : list
            The hidden units for each fully connected layer.
        units_graph_mlp : list
            The hidden units for the layers of the graph feature MLP.
        dropout_rate : float
            Dropout rate.
        use_batchnorm : bool
            If batch normalization should be applied.
        use_residual : bool
            If the block should implement an additive skip connection. If True, this will only be implemented
            if num_input_features == num_output_features for a graph convolution layer.
        """
        super().__init__(num_input_features_gcn, units_graph_convolutions=units_graph_convolutions, 
            units_fully_connected=units_fully_connected, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm, use_residual=use_residual)
        self.layers_graph_mlp = torch.nn.ModuleList()
        for idx, (D, D_prime) in enumerate(zip([num_input_features_graph_mlp] + units_graph_mlp[:-1], units_graph_mlp)):
            is_last_layer = idx == len(units_graph_mlp) - 1
            self.layers_graph_mlp.append(nn.Linear(D, D_prime, bias=True))
            if not is_last_layer and use_batchnorm:
                self.layers_graph_mlp.append(nn.BatchNorm1d(D_prime))
            self.layers_graph_mlp.append(nn.ReLU())
            if not is_last_layer and dropout_rate:
                self.layers_graph_mlp.append(nn.Dropout(dropout_rate))
    

    def forward(self, X, D, masks, F):
        """ Forward pass.
        
        Parameters:
        -----------
        X : torch.FloatTensor, shape [batch_size, N, D]
            The node features.
        D : torch.FloatTensor, shape [batch_size, N, N]
            Pairwise distances for all nodes.
        masks : torch.FloatTensor, shape [batch_size, N, N]
            Masks for the adjacency / distance matrix.
        F : torch.FloatTensor, shape [batch_size, K]
            Graph features that are fed into a separate MLP network.
        """

        # Graph convolutions
        A = self.kernel(D, masks)
        for graph_convolution in self.graph_convolutions:
            X = graph_convolution(X, A, masks)

        # Average pooling
        X = padded_vertex_mean(X, masks, vertex_axis=-2, keepdim=False)

        # Graph feature MLP
        for layer in self.layers_graph_mlp:
            F = layer(F)
        
        # Concatenate graph embeddings with graph feature embeddings
        X = torch.cat([X, F], -1)

        # Fully connecteds, (usually only 1 layer, i.e. logistic regression)
        for layer in self.layers_fully_connected:
            X = layer(X)
        return X
    
