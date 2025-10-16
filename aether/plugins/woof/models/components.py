import math

import torch
import torch.nn as nn


def initialize_convnet_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0, max_pos: int = 5000):
        """
        Position encoding.
        
        Parameters
        ----------
        d_model: int
            The dimension of the embedding.
        dropout: float
            The dropout rate.
        max_pos: int
            The maximum number of positions in the sequence. This will determine the
            size of the pre-computed encoding buffer.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_pos).unsqueeze(1)
        # div_term = 1 / (10000 ** (2 * i / d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_pos, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        # If d_model is odd, the number of cos terms should be 1 less than sin terms.
        end = -1 if d_model % 2 == 1 else None
        pe[:, 1::2] = torch.cos(position * div_term)[:, :end]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.
        
        Parameters
        ----------
        x: torch.Tensor
            A (n, seq_len, embedding_dim) tensor.

        Returns
        -------
        torch.Tensor
            A (n, seq_len, embedding_dim) tensor.
        """
        x = x + self.pe[:x.size(1)]
        if self.dropout.p > 0:
            x = self.dropout(x)
        return x


class Patchifier:
    def __init__(self, patch_size: int):
        self.patch_size = patch_size
        self.padding = [0, 0]
        self.shape_before_flattening = None
        self.patch_grid = None
        
    def patchify(self, image: torch.Tensor, flatten_spatial_dims: bool = True) -> torch.Tensor:
        """Convert a stack of (n_slices, n_channels, h, w) to a stack of
        (n_patches, n_slices, n_channels, patch_size, patch_size) or
        (n_patches, n_slices, -1). If the image size is not divisible by the patch size,
        it will be padded with zeros.
        
        Parameters
        ----------
        image : torch.Tensor
            A stack of (n_slices, n_channels, h, w)
        flatten_spatial_dims : bool, optional
            If True, the spatial and channel dimensions are flattened into a single dimension.
            By default, True.
            
        Returns
        -------
        torch.Tensor
            A stack of (n_patches, n_slices, n_channels, patch_size, patch_size) or 
            (n_patches, n_slices, -1).
        """
        n_slices, n_channels, h, w = image.shape
        
        if h % self.patch_size != 0:
            self.padding[0] = self.patch_size - (h % self.patch_size)
        if w % self.patch_size != 0:
            self.padding[1] = self.patch_size - (w % self.patch_size)
            
        image = torch.nn.functional.pad(image, (0, self.padding[1], 0, self.padding[0]), mode="constant", value=0)
        
        self.patch_grid = (math.ceil(h / self.patch_size), math.ceil(w / self.patch_size))
        
        patches = image.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.reshape(n_slices, n_channels, -1, self.patch_size, self.patch_size)
        patches = patches.permute(2, 0, 1, 3, 4)
        
        self.shape_before_flattening = patches.shape
        
        if flatten_spatial_dims:
            patches = patches.reshape(patches.shape[0], patches.shape[1], -1)
        
        return patches
    
    def assemble(self, patches: torch.Tensor) -> torch.Tensor:
        """Assemble a stack of (n_patches, n_slices, n_channels, patch_size, patch_size)
        or (n_patches, n_slices, -1) into a stack of (n_slices, n_channels, h, w).
        """
        if self.shape_before_flattening is None:
            raise ValueError("assemble can only be called after patchify.")
        
        if patches.ndim == 3:
            patches = patches.reshape(self.shape_before_flattening)
        
        rows = []
        n_cols = self.patch_grid[1]
        for row in range(self.patch_grid[0]):
            image_row = torch.cat(
                [patches[row * n_cols + col] for col in range(n_cols)],
                dim=-1
            )
            rows.append(image_row)
        image = torch.cat(rows, dim=-2)
        
        if self.padding[0] > 0:
            image = image[..., :-self.padding[0], :]
        if self.padding[1] > 0:
            image = image[..., :-self.padding[1]]
        
        return image
