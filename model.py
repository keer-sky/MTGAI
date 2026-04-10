import numpy as np
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class FixedTaskInteractionModule(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1, num_classes=None):
        super(FixedTaskInteractionModule, self).__init__()
        self.d_model = d_model

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.reg_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.cls_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, reg_features, cls_features):
        batch_size = reg_features.size(0)

        reg_features_seq = reg_features.unsqueeze(1)
        cls_features_seq = cls_features.unsqueeze(1)

        reg_enhanced, _ = self.cross_attention(
            query=reg_features_seq,
            key=cls_features_seq,
            value=cls_features_seq
        )

        cls_enhanced, _ = self.cross_attention(
            query=cls_features_seq,
            key=reg_features_seq,
            value=reg_features_seq
        )

        reg_enhanced = reg_enhanced.squeeze(1)
        cls_enhanced = cls_enhanced.squeeze(1)

        fused_features = self.feature_fusion(
            torch.cat([reg_enhanced, cls_enhanced], dim=-1)
        )

        reg_gate = self.reg_gate(fused_features)
        cls_gate = self.cls_gate(fused_features)

        reg_final = reg_features + reg_gate * fused_features
        cls_final = cls_features + cls_gate * fused_features

        return reg_final, cls_final


class FixedFocusClassTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=128, nhead=8, num_layers=4,
                 num_classes=2, prediction_length=1, dropout=0.2, focus_classes=None):
        super(FixedFocusClassTransformer, self).__init__()

        self.d_model = d_model
        self.prediction_length = prediction_length
        self.num_classes = num_classes
        self.focus_classes = focus_classes

        self.conv_embedding = nn.Sequential(
            nn.Conv1d(input_dim, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        self.input_projection = nn.Linear(d_model, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.reg_attention_pool = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cls_attention_pool = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.reg_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.cls_query = nn.Parameter(torch.randn(1, 1, d_model))

        self.task_interaction = FixedTaskInteractionModule(d_model, nhead, dropout, num_classes)

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, prediction_length)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, mean=0.0, std=0.02)

    def forward(self, x, class_labels=None):
        batch_size, seq_len, input_dim = x.shape

        x_conv = x.transpose(1, 2)
        x_conv = self.conv_embedding(x_conv)
        x_conv = x_conv.transpose(1, 2)

        x_proj = self.input_projection(x_conv) * np.sqrt(self.d_model)

        x_pos = x_proj.transpose(0, 1)
        x_pos = self.pos_encoder(x_pos)
        x_pos = x_pos.transpose(0, 1)

        encoded = self.transformer_encoder(x_pos)

        reg_query = self.reg_query.expand(batch_size, -1, -1)
        cls_query = self.cls_query.expand(batch_size, -1, -1)

        reg_features, _ = self.reg_attention_pool(reg_query, encoded, encoded)
        cls_features, _ = self.cls_attention_pool(cls_query, encoded, encoded)

        reg_features = reg_features.squeeze(1)
        cls_features = cls_features.squeeze(1)

        reg_enhanced, cls_enhanced = self.task_interaction(reg_features, cls_features)

        regression_output = self.regression_head(reg_enhanced)
        classification_output = self.classification_head(cls_enhanced)

        return regression_output, classification_output