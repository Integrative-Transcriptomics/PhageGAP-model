"""Model definitions for protein category classification."""

from __future__ import annotations

from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import math
import optuna


class MLP(nn.Module):
    """Small MLP for protein category classification

    Parameters
    ----------
    in_dim (int): Input dimension, must match pLM embedding dimension
    num_dimensions (int): Number of hidden dimensions
    num_neurons (List[int]): Number of neurons of hidden dimensions.
    dropout (float): Dropout probability applied after each ReLU (0 -> disabled)
    num_classes (int): Number of target classes
    """
    def __init__(
        self,
        trial: optuna.trial._trial.Trial,
        in_dim: int,
        num_dimensions: int,
        num_neurons: List[int],
        dropout: float,
        num_classes: int,
        n_feats: int,
    ) -> None:
        super().__init__()

        assert in_dim in (1024, 960, 1280, 1536, 800, 1280), "in_dim must match embedding dimension"
        assert num_classes >= 2, "Need at least two classes"
        assert 0.0 <= dropout <= 1.0, "dropout outside [0,1]"
        assert len(num_neurons) > 0, "Provide at least one hidden dimension"
        assert len(num_neurons) == num_dimensions, "Number of dimensions and length of nun_neurons list must match"

        self.n_feats = n_feats
        self.layers: List[nn.Module] = []

        prev_channel = in_dim + n_feats
        for i in range(num_dimensions):
            self.layers.extend([
                nn.Linear(prev_channel, num_neurons[i]),
                nn.BatchNorm1d(num_neurons[i]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_channel = num_neurons[i]

        self.layers.append(nn.Linear(prev_channel, num_classes))
        self.net = nn.Sequential(*self.layers)


    def forward(self, x, feats: torch.Tensor | None = None):
        x = F.normalize(x, dim=1)

        if feats is not None:
            x = torch.cat([x, feats], dim=1)

        return self.net(x)
   

class CNN(nn.Module):
    """Improved CNN with dilations for protein category classification

    Parameters
    ----------
    in_channels (int): Number of input channels
    num_conv_layers (int): Number of convolutional layers
    num_filters (List[int]): Number of filters of convolutional layers
    num_neurons (int): Number of neurons of FC layers
    kernel_sizes (List[int]): Kernel sizes for conv layers
    dropout (float): Dropout probability applied after each ReLU (0 -> disabled)
    num_classes (int): Number of target classes
    dilations (List[int]): Dilation factors for each convolutional block
    use_dilation (bool):    If True, apply defined dilations
                            If False, all convolutions have dilation = 1
    """

    def __init__(
        self,
        trial: optuna.trial._trial.Trial,
        in_channels: int,
        num_conv_layers: int,
        num_filters: List[int],
        kernel_sizes: List[int],
        dropout: float,
        num_classes: int,
        dilations: List[int],
        use_dilation: bool,
        mean_max: bool,
        n_feats: int,
        use_linear_attention: bool,
        use_nonlinear_attention: bool,
    ) -> None:
        super().__init__()
        assert in_channels in (1024, 960, 1280, 1536, 640, 2560, 1152), "in_channels must match embedding dimension"
        assert num_classes >= 2, "Need at least two classes"
        assert all(k % 2 == 1 for k in kernel_sizes), "all kernel_sizes should be odd for symmetric padding" # to keep sequence length stable
        assert 0.0 <= dropout <= 1.0, "dropout outside [0,1]"
        assert len(num_filters) > 0, "Provide at least one conv layer"
        assert len(num_filters) == len(kernel_sizes) == len(dilations), "conv_channels, kernel_sizes, and dilations lists must have same length"
        assert not (use_linear_attention and use_nonlinear_attention), "Can't use both linear and nonlinear attention, please decide"

        self.mean_max = mean_max

        layers: List[nn.Module] = []

        prev_channels = in_channels

        for i in range(num_conv_layers):
            d = dilations[i] if use_dilation else 1
            padding = (kernel_sizes[i] // 2) * d
            layers.append(
                nn.Conv1d(
                    in_channels = prev_channels,
                    out_channels = num_filters[i],
                    kernel_size = kernel_sizes[i],
                    padding = padding,
                    dilation = d
                )
            )
            layers.append(nn.BatchNorm1d(num_filters[i]))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_channels = num_filters[i]

        self.cnn = nn.Sequential(*layers)
        self.use_linear_attention = use_linear_attention
        self.use_nonlinear_attention = use_nonlinear_attention

        if use_linear_attention:
            self.pool = AttentionPooling(prev_channels)
        elif use_nonlinear_attention:
            self.pool = NonlinearAttentionPooling(prev_channels)

        # after pooling, map final feature vector to class logits
        if mean_max:
            self.classifier = nn.Linear(2 * prev_channels+n_feats, num_classes)
        else:
            self.classifier = nn.Linear(prev_channels+n_feats, num_classes)


    def forward(self, x: torch.Tensor, mask: torch.Tensor, feats: torch.Tensor, return_features=False) -> torch.Tensor:  # noqa: D401
        """
        x: (B, L, D)
        mask: (B, L) with 1=real, 0=padding
        feats: (B, n_feats)
        """
        assert x.ndim == 3, "Expected input shape (batch, seq_len, embed_dim)"
        x = x.permute(0, 2, 1) # from x = (B, L, D) to x = (B, D, L)

        # apply CNN -> each position in sequence has C learned feature channels
        fts = self.cnn(x) # (B, C, L)

        if self.use_linear_attention or self.use_nonlinear_attention:
            pooled = self.pool(fts, mask) # (B, C)
        else:
            # masked global average pool
            pooled = masked_mean(fts, mask) # (B, C)
            if self.mean_max:
                max_pooled = masked_max(fts, mask)
                pooled = torch.cat([pooled, max_pooled], dim=1)

        if feats is not None:
            combined = torch.cat([pooled, feats], dim=1)  # (B, C + n_feats)
            logits =  self.classifier(combined) # (B, num_classes)
            if return_features:
                return logits, combined
    
            return logits
        else:
            logits = self.classifier(pooled) # (B, num_classes)
            if return_features:
                return logits, pooled
     
            return logits
    


class CNN_Flattened(nn.Module):
    """CNN with dilations flattened for protein category classification

    Parameters
    ----------
    in_channels (int): Number of input channels
    num_conv_layers (int): Number of convolutional layers
    num_filters (List[int]): Number of filters of convolutional layers
    num_neurons (int): Number of neurons of FC layers
    kernel_sizes (List[int]): Kernel sizes for conv layers
    dropout (float): Dropout probability applied after each ReLU (0 -> disabled)
    num_classes (int): Number of target classes
    dilations (List[int]): Dilation factors for each convolutional block
    use_dilation (bool):    If True, apply defined dilations
                            If False, all convolutions have dilation = 1
    """

    def __init__(
        self,
        trial: optuna.trial._trial.Trial,
        in_channels: int,
        num_conv_layers: int,
        num_filters: List[int],
        kernel_sizes: List[int],
        dropout: float,
        num_classes: int,
        dilations: List[int],
        use_dilation: bool,
        mean_max: bool,
        n_feats: int,
    ) -> None:
        super().__init__()
        assert in_channels in (1024, 960, 1280, 1536, 640, 2560, 1152), "in_channels must match embedding dimension"
        assert num_classes >= 2, "Need at least two classes"
        assert all(k % 2 == 1 for k in kernel_sizes), "all kernel_sizes should be odd for symmetric padding" # to keep sequence length stable
        assert 0.0 <= dropout <= 1.0, "dropout outside [0,1]"
        assert len(num_filters) > 0, "Provide at least one conv layer"
        assert len(num_filters) == len(kernel_sizes) == len(dilations), "conv_channels, kernel_sizes, and dilations lists must have same length"

        self.mean_max = mean_max
        self.blocks = nn.ModuleList()

        prev_channels = in_channels

        for i in range(num_conv_layers):
            d = dilations[i] if use_dilation else 1
            padding = (kernel_sizes[i] // 2) * d
            block = nn.Sequential(
                nn.Conv1d(
                    in_channels = prev_channels,
                    out_channels = num_filters[i],
                    kernel_size = kernel_sizes[i],
                    padding = padding,
                    dilation = d
                ),
                nn.BatchNorm1d(num_filters[i]),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )
            self.blocks.append(block)
            prev_channels = num_filters[i]

        # after pooling, map final feature vector to class logits
        if mean_max:
            self.classifier = nn.Linear(2 * sum(num_filters)+n_feats, num_classes)
        else:
            self.classifier = nn.Linear(sum(num_filters)+n_feats, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, features: torch.Tensor | None = None) -> torch.Tensor:  # noqa: D401
        """
        x: (B, L, D)
        mask: (B, L) with 1=real, 0=padding
        feats: (B, n_feats)
        """
        assert x.ndim == 3, "Expected input shape (batch, seq_len, embed_dim)"
        x = x.permute(0, 2, 1) # from x = (B, L, D) to x = (B, D, L)

        pooled_outputs = []

        for block in self.blocks:
            x = block(x) # (B, C, L)

            # masked global average pool
            pooled = masked_mean(x, mask) # (B, C)
            if self.mean_max:
                max_pooled = masked_max(x, mask)
                pooled = torch.cat([pooled, max_pooled], dim=1)

            pooled_outputs.append(pooled)

        flattened = torch.cat(pooled_outputs, dim=1) # (B, sum(num_filters))

        if features is not None:
            combined = torch.cat([flattened, features], dim=1)  # (B, C + n_feats)
            return self.classifier(combined) # (B, num_classes)
        else:
            return self.classifier(flattened) # (B, num_classes)



class CNN_MLP(nn.Module):
    """CNN with dilations for protein category classification, subsequent MLP as classification head

    Parameters
    ----------
    in_channels (int): Number of input channels
    num_conv_layers (int): Number of convolutional layers
    num_filters (List[int]): Number of filters of convolutional layers
    num_neurons (int): Number of neurons of FC layers
    kernel_sizes (List[int]): Kernel sizes for conv layers
    dropout_conv (float): Dropout probability applied after each ReLU (0 -> disabled)
    num_classes (int): Number of target classes
    dilations (List[int]): Dilation factors for each convolutional block
    use_dilation (bool):    If True, apply defined dilations
                            If False, all convolutions have dilation = 1
    num_dimensions (int): Number of hidden dimensions
    num_neurons (List[int]): Number of neurons of hidden dimensions.
    dropout_mlp (float): Dropout probability applied after each ReLU (0 -> disabled)
    """

    def __init__(
        self,
        trial: optuna.trial._trial.Trial,
        in_channels: int,
        num_conv_layers: int,
        num_filters: List[int],
        kernel_sizes: List[int],
        dropout_conv: float,
        num_classes: int,
        dilations: List[int],
        use_dilation: bool,
        use_linear_attention: bool,
        use_nonlinear_attention: bool,
        num_dimensions: int,
        num_neurons: List[int],
        dropout_mlp: float,
        n_feats: int,
    ) -> None:
        super().__init__()
        assert in_channels in (1024, 960, 1280, 1536, 800), "in_channels must match embedding dimension"
        assert num_classes >= 2, "Need at least two classes"
        assert all(k % 2 == 1 for k in kernel_sizes), "all kernel_sizes should be odd for symmetric padding" # to keep sequence length stable
        assert (0.0 <= dropout_conv <= 1.0) & (0.0 <= dropout_mlp <= 1.0), "dropout outside [0,1]"
        assert len(num_filters) > 0, "Provide at least one conv layer"
        assert len(num_filters) == len(kernel_sizes) == len(dilations), "conv_channels, kernel_sizes, and dilations lists must have same length"
        assert not (use_linear_attention and use_nonlinear_attention), "Can't use both linear and nonlinear attention, please decide"
        assert len(num_neurons) > 0, "Provide at least one hidden dimension"
        assert len(num_neurons) == num_dimensions, "Number of dimensions and length of nun_neurons list must match"

        self.n_feats = n_feats
        self.cnn_layers: List[nn.Module] = []

        prev_channels = in_channels

        for i in range(num_conv_layers):
            d = dilations[i] if use_dilation else 1
            padding = (kernel_sizes[i] // 2) * d
            self.cnn_layers.append(
                nn.Conv1d(
                    in_channels = prev_channels,
                    out_channels = num_filters[i],
                    kernel_size = kernel_sizes[i],
                    padding = padding,
                    dilation = d
                )
            )
            self.cnn_layers.append(nn.BatchNorm1d(num_filters[i]))
            self.cnn_layers.append(nn.ReLU())
            if dropout_conv > 0.0:
                self.cnn_layers.append(nn.Dropout(p=dropout_conv))
            prev_channels = num_filters[i]

        self.cnn = nn.Sequential(*self.cnn_layers)
        self.use_linear_attention = use_linear_attention
        self.use_nonlinear_attention = use_nonlinear_attention

        if use_linear_attention:
            self.pool = AttentionPooling(prev_channels)
        elif use_nonlinear_attention:
            self.pool = NonlinearAttentionPooling(prev_channels)

        # after CNN and pooling, map final feature vector to class logits
        prev_channels = prev_channels + n_feats
        self.mlp_layers: List[nn.Module] = []
        for i in range(num_dimensions):
            self.mlp_layers.extend([
                nn.Linear(prev_channels, num_neurons[i]),
                nn.BatchNorm1d(num_neurons[i]),
                nn.ReLU(),
                nn.Dropout(dropout_mlp)
            ])
            prev_channels = num_neurons[i]

        self.mlp_layers.append(nn.Linear(prev_channels, num_classes))
        self.classifier = nn.Sequential(*self.mlp_layers)


    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, feats: torch.Tensor | None = None) -> torch.Tensor:  # noqa: D401
        """
        x: (B, L, D)
        mask: (B, L) with 1=real, 0=padding
        """
        assert x.ndim == 3, "Expected input shape (batch, seq_len, embed_dim)"
        x = x.permute(0, 2, 1) # from x = (B, L, D) to x = (B, D, L)

        # apply CNN -> each position in sequence has C learned feature channels
        fts = self.cnn(x) # (B, C, L)

        if self.use_linear_attention or self.use_nonlinear_attention:
            pooled = self.pool(fts, mask) # (B, C)
        else:
            # masked global average pool
            pooled = masked_mean(fts, mask) # (B, C)

        pooled = F.normalize(pooled, dim=1)

        if feats is not None:
            pooled = torch.cat([pooled, feats], dim=1)
        return self.classifier(pooled) # (B, num_classes)


def masked_mean(feats, mask): # feats (B, C, L), mask (B, L)
    """mean-pools over sequence length, ignoring padded positions"""
    mask = mask.unsqueeze(1) # (B, 1, L)
    # multiply features by mask -> ignore padded positions
    feats = feats * mask
    # average over sequence length -> global average pooling over valid residues only
    pooled = feats.sum(dim=2) / mask.sum(dim=2).clamp(min=1)
    return pooled


def masked_max(feats, mask): # feats (B, C, L), mask (B, L)
    """max-pools over sequence length, ignoring padded positions"""
    mask = mask.unsqueeze(1) # (B, 1, L)
    # set masked positions to -inf -> ignore padded positions
    feats = feats.masked_fill(mask == 0, float("-inf"))
    # average over sequence length -> global max pooling over valid residues only
    pooled = feats.max(dim=2).values 
    return pooled


class AttentionPooling(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # maps feature vector at each position (C) -> scalar score
        self.score = nn.Linear(channels, 1)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor | None = None, features: torch.Tensor | None = None) -> torch.Tensor:
        """
        feats: (B, C, L) 
        mask: (B, L) with 1=real, 0=padding
        returns: (B, C) pooled protein representation
        """
        feats = feats.permute(0, 2, 1) # (B, C, L) -> (B, L, C)
        scores = self.score(feats) # (B, L, 1), raw attention scores per position

        # mask padding positions -> set to -inf
        scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))

        # normalize across sequence length
        attention = torch.softmax(scores, dim=1) # (B, L, 1)

        # weighted sum over L
        pooled = torch.sum(attention * feats, dim=1) # (B, C)

        return pooled


class NonlinearAttentionPooling(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # maps feature vector at each position (C) -> scalar score
        self.score = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.Tanh(),
            nn.Linear(channels // 2, 1)
        )

    def forward(self, feats: torch.Tensor, mask: torch.Tensor | None = None, features: torch.Tensor | None = None) -> torch.Tensor:
        """
        feats: (B, C, L) 
        mask: (B, L) with 1=real, 0=padding
        returns: (B, C) pooled protein representation
        """
        feats = feats.permute(0, 2, 1) # (B, C, L) -> (B, L, C)
        scores = self.score(feats) # (B, L, 1), raw attention scores per position

        # mask padding positions -> set to -inf
        scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))

        # normalize across sequence length
        attention = torch.softmax(scores, dim=1) # (B, L, 1)

        # weighted sum over L
        pooled = torch.sum(attention * feats, dim=1) # (B, C)

        return pooled
    





















    

class CNNTransformer(nn.Module):
    """CNN + Transformer for protein category classification

    Parameters
    ----------
    in_channels (int): Number of input channels
    conv_channels (List[int]): Output channels for each Conv1d block. Length defines depth
    kernel_sizes (List[int]): Kernel sizes for each conv layer (must all be odd)
    dropout (float): Dropout probability applied after each ReLU (0 -> disabled)
    num_classes (int): Number of target classes
    dilations (List[int]): Dilation factors for each convolutional block
    use_dilation (bool):    If True, apply defined dilations (preferentially exponentially increasing (1, 2, 4, ...))
                            If False, all convolutions have dilation = 1
    num_heads (int): Number of attention heads in Transformer
    num_transformer_layers (int): Number of Transformer encoder layers
    num_hidden_layers (int): Number of hidden layers
    """

    def __init__(
        self,
        in_channels: int,
        conv_channels: List[int],
        kernel_sizes: List[int],
        dropout: float,
        num_classes: int,
        dilations: List[int],
        use_dilation: bool,
        num_heads: int,
        num_transformer_layers: int,
        hidden_layers: int,
        use_linear_attention: bool,
        use_nonlinear_attention: bool,
    ) -> None:
        super().__init__()
        assert (in_channels == 1024 or in_channels == 960), "in_channels must match embedding dimension (either 1024 [prot_t5] or 960 [esmc_300m])"
        assert num_classes >= 2, "Need at least two classes"
        assert all(k % 2 == 1 for k in kernel_sizes), "all kernel_sizes should be odd for symmetric padding" # to keep sequence length stable
        assert 0.0 <= dropout <= 1.0, "dropout outside [0,1]"
        assert len(conv_channels) > 0, "Provide at least one conv layer"
        assert len(conv_channels) == len(kernel_sizes) == len(dilations), "conv_channels, kernel_sizes, and dilations lists must have same length"
        assert num_heads > 0, "Number of heads must be positive"
        assert num_transformer_layers > 0, "Number of transformer layers must be positive"
        assert not (use_linear_attention and use_nonlinear_attention), "Can't use both linear and nonlinear attention, please decide"

        layers: List[nn.Module] = []

        prev_channels = in_channels
        dilation = 0

        for out_channels, k in zip(conv_channels, kernel_sizes): # multiple convs of different kernel sizes to capture short motifs and long-range patterns
            d = dilations[dilation] if use_dilation else 1
            padding = (k // 2) * d
            layers.append(
                nn.Conv1d(
                    in_channels = prev_channels,
                    out_channels = out_channels,
                    kernel_size = k,
                    padding = padding,
                    dilation = d
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_channels = out_channels

            if use_dilation:
                dilation += 1

        self.cnn = nn.Sequential(*layers)
        self.use_linear_attention = use_linear_attention
        self.use_nonlinear_attention = use_nonlinear_attention

        embed_dim = prev_channels # embed_dim equal to number of out_channels for last convolution
        if use_linear_attention:
            self.pool = AttentionPooling(embed_dim)
        elif use_nonlinear_attention:
            self.pool = NonlinearAttentionPooling(embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = embed_dim,
            nhead = num_heads,
            dim_feedforward = 4 * embed_dim,
            dropout = dropout,
            activation = "gelu",
            batch_first = True, # (B, L, C)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_layers),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_layers, num_classes)
        )
        

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """
        x: (B, L, D)
        mask: (B, L) with 1=real, 0=padding
        """
        assert x.ndim == 3, "Expected input shape (batch, seq_len, embed_dim)"
        x = x.permute(0, 2, 1) # from x = (B, L, D) to x = (B, D, L)

        # apply CNN -> each position in sequence has C learned feature channels
        feats = self.cnn(x) # (B, C, L)
        feats = feats.permute(0, 2, 1) # (B, L, C) prepare for transformer

        # Transformer
        key_padding_mask = (mask == 0) # (B, L), Transformer expects: True = ignore position
        feats = self.transformer(feats, src_key_padding_mask=key_padding_mask) # (B, L, C)
        feats = feats.permute(0, 2, 1) # (B, C, L) prepare for masked_mean

        if self.use_linear_attention or self.use_nonlinear_attention:
            pooled = self.pool(feats, mask) # (B, C)
        else:
            # masked global average pool
            pooled = masked_mean(feats, mask) # (B, C)

        return self.classifier(pooled) # (B, num_classes)
    


class Transformer(nn.Module):
    """Transformer for protein category classification

    Parameters
    ----------
    in_channels (int): Number of input channels
    conv_channels (List[int]): Output channels for each Conv1d block. Length defines depth
    kernel_sizes (List[int]): Kernel sizes for each conv layer (must all be odd)
    dropout (float): Dropout probability applied after each ReLU (0 -> disabled)
    num_classes (int): Number of target classes
    dilations (List[int]): Dilation factors for each convolutional block
    use_dilation (bool):    If True, apply defined dilations (preferentially exponentially increasing (1, 2, 4, ...))
                            If False, all convolutions have dilation = 1
    num_heads (int): Number of attention heads in Transformer
    num_transformer_layers (int): Number of Transformer encoder layers
    num_hidden_layers (int): Number of hidden layers
    """

    def __init__(
        self,
        in_channels: int,
        dropout: float,
        num_classes: int,
        num_heads: int,
        num_transformer_layers: int,
        use_linear_attention: bool,
        use_nonlinear_attention: bool,
    ) -> None:
        super().__init__()
        assert (in_channels == 1024 or in_channels == 960), "in_channels must match embedding dimension (either 1024 [prot_t5] or 960 [esmc_300m])"
        assert num_classes >= 2, "Need at least two classes"
        assert 0.0 <= dropout <= 1.0, "dropout outside [0,1]"
        assert num_heads > 0, "Number of heads must be positive"
        assert num_transformer_layers > 0, "Number of transformer layers must be positive"
        assert not (use_linear_attention and use_nonlinear_attention), "Can't use both linear and nonlinear attention, please decide"

        self.use_linear_attention = use_linear_attention
        self.use_nonlinear_attention = use_nonlinear_attention

        embed_dim = in_channels # embed_dim equal to number of in_channels
        if use_linear_attention:
            self.pool = AttentionPooling(embed_dim)
        elif use_nonlinear_attention:
            self.pool = NonlinearAttentionPooling(embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = embed_dim,
            nhead = num_heads,
            dim_feedforward = 4 * embed_dim,
            dropout = dropout,
            activation = "relu",
            batch_first = True, # (B, L, C)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # classifier
        self.classifier = nn.Linear(embed_dim, num_classes)
        

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """
        x: (B, L, D)
        mask: (B, L) with 1=real, 0=padding
        """
        assert x.ndim == 3, "Expected input shape (batch, seq_len, embed_dim)"
        x = x.permute(0, 2, 1) # from x = (B, L, D) to x = (B, D, L)

        # apply CNN -> each position in sequence has C learned feature channels
        feats = self.cnn(x) # (B, C, L)
        feats = feats.permute(0, 2, 1) # (B, L, C) prepare for transformer

        # Transformer
        key_padding_mask = (mask == 0) # (B, L), Transformer expects: True = ignore position
        feats = self.transformer(feats, src_key_padding_mask=key_padding_mask) # (B, L, C)
        feats = feats.permute(0, 2, 1) # (B, C, L) prepare for masked_mean

        if self.use_linear_attention or self.use_nonlinear_attention:
            pooled = self.pool(feats, mask) # (B, C)
        else:
            # masked global average pool
            pooled = masked_mean(feats, mask) # (B, C)

        return self.classifier(pooled) # (B, num_classes)
    

class PositionalEncoding(nn.Module):

    def __init__(self, 
                 embed_dim: int, 
                 dropout: float,
                 context_size: int
                 ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(context_size, embed_dim)
        position = torch.arange(0, context_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ContextTransformer(nn.Module):
    """
    based on a pytorch TransformerEncoder.
    """

    def __init__(
            self,
            num_heads: int,
            dim_feedforward: int,
            num_layers: int,
            dropout: float,
            num_classes: int,
            embed_dim: int,
            context_size: int
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_classes = num_classes

        # learnable vector for each position, initially random normal distribution (cf. weight initialization)
        self.positional_encoding = nn.Parameter(torch.randn(1, context_size, embed_dim)) # (batch size, context size, embed dim)

        encoder_layer = nn.TransformerEncoderLayer(
            # TransformerEncoderLayer is made up of self-attn and feedforward network.
            d_model = self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True, # input (B, L, D)
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(self.embed_dim, self.num_classes)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask):
        # embedding + position information
        x = x + self.positional_encoding 
        padding_mask = ~mask # Trafo expects True = ignore
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask) # (batch, context_size, embed_dim)

        # masked mean pooling
        mask = mask.unsqueeze(-1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1) # (batch, embed_dim)

        logits = self.classifier(x) # (batch, num_classes)

        return logits
