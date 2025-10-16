from typing import Optional

import torch
import torch.nn as nn

from aether.plugins.woof.models.components import PositionalEncoding


class TransformerLatentProcessor(nn.Module):
    def __init__(
        self,
        num_in_channels: int = 128,
        num_hidden_dims: int = 768,
        num_feedforward_hidden_dims: int = 2048,
        n_layers: int = 4,
        dropout: float = 0,
        num_projected_out_dims: Optional[int] = None,
    ):
        """
        Transformer latent processor.
        
        Parameters
        ----------
        num_in_channels: int
            The number of input channels.
        num_hidden_dims: int
            The number of hidden dimensions.
        num_feedforward_hidden_dims: int
            The number of feedforward hidden dimensions.
        n_layers: int
            The number of layers.
        dropout: float
            The dropout rate.
        num_projected_out_dims: Optional[int]
            The number of projected output channels. If given, the output from
            the transformer with num_hidden_dims channels will be projected to
            num_projected_out_dims channels by a linear layer. Otherwise,
            the output will have num_hidden_dims channels.
        """
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_hidden_dims = num_hidden_dims
        self.num_feedforward_hidden_dims = num_feedforward_hidden_dims
        self.num_projected_out_dims = num_projected_out_dims
        self.n_layers = n_layers
        self.dropout = dropout
        
        self._chunk_size = 1000000

        self.projector = nn.Linear(num_in_channels, num_hidden_dims)
        self.positional_encoder = PositionalEncoding(num_hidden_dims, dropout=0, max_pos=100)
        
        self.transformer = None
        self.build_transformer()
        
        if self.num_projected_out_dims is not None:
            self.projector_out = nn.Linear(num_hidden_dims, num_projected_out_dims)
        else:
            self.projector_out = None
        
    def build_transformer(self):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.num_hidden_dims,
            nhead=8,
            dim_feedforward=self.num_feedforward_hidden_dims,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
        )
    
    def forward(self, x):
        """Forward pass through the transformer.
        
        Parameters
        ----------
        x: torch.Tensor
            A (n, seq_len, num_in_channels) tensor.

        Returns
        -------
        torch.Tensor
            A (n, seq_len, num_hidden_dims or num_projected_out_dims) tensor.
        """
        # Chunk x and process sequentially to save memory.
        x = torch.chunk(x, max(1, int(x.shape[0] / self._chunk_size)), dim=0)
        x = [self.forward_chunk(x_chunk) for x_chunk in x]
        x = torch.cat(x, dim=0)
        return x
    
    def forward_chunk(self, x):
        x = self.projector(x)
        x = self.positional_encoder(x)
        x = self.transformer(x)
        if self.projector_out is not None:
            x = self.projector_out(x)
        return x
