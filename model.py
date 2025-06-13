# model.py
import torch.nn as nn

class RNN_GRU(nn.Module):
    def __init__(self, input_dim, num_classes,
                 gru_hidden_size=320, num_layers=2,
                 attention_heads=4, dense_size=512,
                 dropout=0.3, bidirectional=True,
                 activation_fn='mish'):
        super().__init__()
        self.activation = (
            nn.ReLU() if activation_fn == 'relu'
            else nn.GELU() if activation_fn == 'gelu'
            else nn.Mish()
        )
        self.gru = nn.GRU(input_dim, gru_hidden_size, num_layers=num_layers,
                          bidirectional=bidirectional, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.MultiheadAttention(
            embed_dim=gru_hidden_size * (2 if bidirectional else 1),
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size * (2 if bidirectional else 1), dense_size),
            self.activation,
            nn.LayerNorm(dense_size),
            nn.Dropout(dropout),
            nn.Linear(dense_size, num_classes)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        pooled = attn_out.mean(dim=1)
        return self.classifier(pooled)
