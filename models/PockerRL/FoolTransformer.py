import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerNoMaskNoPos(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048):
        super().__init__()
        self.d_model = d_model

        # Общий эмбеддинг
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Позиционное кодирование только для энкодера
        self.pos_encoder = PositionalEncoding(d_model)

        # Трансформер (стандартный, но маски и позиционное кодирование уберем позже)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )

        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # Эмбеддинг + позиционное кодирование ТОЛЬКО для энкодера
        src_emb = self.embedding(src) * (self.d_model ** 0.5)
        src_emb = self.pos_encoder(src_emb)

        # tgt без позиционного кодирования
        tgt_emb = self.embedding(tgt) * (self.d_model ** 0.5)

        # Без маскировки
        output = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=None,  # не маскируем энкодер
            tgt_mask=None,  # не маскируем декодер
            memory_mask=None,
            src_key_padding_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None
        )

        return self.output_layer(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x