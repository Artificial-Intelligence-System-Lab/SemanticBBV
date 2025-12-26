import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import (
    SetAttentionBlock,
    InducedSetAttentionBlock,
    PoolingMultiheadAttention,
)

class SetTransformerWithWeights(nn.Module):
    def __init__(self, feature_dim, D=128, m=16, h=4, k=4):
        """
        feature_dim: Original feature dimension d for each point
        We will map both features + scalar weights to D dimensions, then add -> D dimensions
        Finally k*D = 512
        """
        super().__init__()
        # —— two parallel embeddings —— 
        self.embed_x = nn.Linear(feature_dim, D)
        self.embed_w = nn.Linear(1, D)

        # —— encoder / decoder almost identical to the original —— 
        self.encoder = nn.Sequential(
            InducedSetAttentionBlock(D, m, h, RFF(D), RFF(D)),
            InducedSetAttentionBlock(D, m, h, RFF(D), RFF(D))
        )
        self.decoder = nn.Sequential(
            PoolingMultiheadAttention(D, k, h, RFF(D)),
            SetAttentionBlock(D, h, RFF(D))
        )

        # The output we want is directly b×(k*D)=b×512
        # Can use identity, or not write predictor
        self.to_embedding = nn.Identity()

        # Weight initialization
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(weights_init)

    def forward(self, x, w):
        """
        x: float tensor, shape [b, n, d]
        w: float tensor, shape [b, n, 1]
        returns:
          z: float tensor, shape [b, 512] —— embedding of the entire set
        """
        # 1) Embed separately, then add and activate
        h_x = self.embed_x(x)      # [b, n, D]
        h_w = self.embed_w(w)      # [b, n, D]
        h = F.relu(h_x + h_w)      # [b, n, D]

        # 2) Set Transformer encoding + decoding
        h = self.encoder(h)        # [b, n, D]
        h = self.decoder(h)        # [b, k, D]

        # 3) Flatten to b×(k*D)=b×512
        b, k, D = h.shape
        z = h.contiguous().view(b, k * D)

        # 4) (Optional) linear mapping/normalization...
        return self.to_embedding(z)

class RFF(nn.Module):
    """
    Row-wise FeedForward layers.
    """
    def __init__(self, d):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(inplace=True),
            nn.Linear(d, d), nn.ReLU(inplace=True),
            nn.Linear(d, d), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        """
        x: [b, n, d] -> out: [b, n, d]
        """
        # Here PyTorch automatically treats the last dimension as the feature dimension for linear
        return self.layers(x)