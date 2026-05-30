import torch
import torch.nn as nn
import torch.nn.functional as F
from esm.modules import TransformerLayer, ESM1bLayerNorm, RobertaLMHead
import esm


class ConvUnetAttentionLayer(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, attention_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.pre_norm = ESM1bLayerNorm(embed_dim)
        self.activation = nn.GELU()

        # Encoder Path
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=3)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck_transformer = TransformerLayer(
            embed_dim=embed_dim, ffn_embed_dim=ffn_embed_dim, attention_heads=attention_heads,
            add_bias_kv=False, use_esm1b_layer_norm=True, use_rotary_embeddings=True,
        )

        # Decoder Path
        self.unpool1 = nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=2, stride=2)
        self.tconv1 = nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=7, padding=3)
        self.norm_t1 = nn.LayerNorm(embed_dim)
        self.unpool2 = nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=5, padding=2)
        self.norm_t2 = nn.LayerNorm(embed_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def _apply_norm(self, norm_layer, x_conv):
        x_permuted = x_conv.permute(0, 2, 1)
        x_normed = norm_layer(x_permuted)
        return x_normed.permute(0, 2, 1)

    def forward(self, x, self_attn_padding_mask=None, need_head_weights=False, return_bottleneck_only=False):
        # x: [T, B, C]
        x_residual = x
        x_conv = self.pre_norm(x).permute(1, 2, 0)

        h_conv1 = self.conv1(x_conv)
        skip1 = self.activation(self._apply_norm(self.norm1, h_conv1))
        h_pool1 = self.pool1(skip1)
        h_conv2 = self.conv2(h_pool1)
        skip2 = self.activation(self._apply_norm(self.norm2, h_conv2))
        h_pool2 = self.pool2(skip2)

        # Transformer Bottleneck
        h_attn_in = h_pool2.permute(2, 0, 1)
        pooled_padding_mask = None
        if self_attn_padding_mask is not None:
            m = self_attn_padding_mask.float().unsqueeze(1)
            pooled_mask_1 = F.max_pool1d(m, kernel_size=2, stride=2)
            pooled_mask_2 = F.max_pool1d(pooled_mask_1, kernel_size=2, stride=2).bool()
            pooled_padding_mask = pooled_mask_2.squeeze(1)

        # [FIX] Explicitly pass need_head_weights to the underlying TransformerLayer
        attn_out, attn_weights = self.bottleneck_transformer(
            h_attn_in,
            self_attn_padding_mask=pooled_padding_mask,
            need_head_weights=need_head_weights
        )

        if return_bottleneck_only:
            return attn_out.permute(1, 2, 0)  # Return [B, C, L_compressed]

        # Decoder / Upsample
        h_upsample_in = attn_out.permute(1, 2, 0)
        h_up1 = self.unpool1(h_upsample_in)
        h_up1_tconv = self.tconv1(h_up1 + skip2)
        h_up1 = self.activation(self._apply_norm(self.norm_t1, h_up1_tconv))

        h_up2 = self.unpool2(h_up1)
        h_up2_tconv = self.tconv2(h_up2 + skip1)
        h_up2 = self.activation(self._apply_norm(self.norm_t2, h_up2_tconv))

        x_out = h_up2.permute(2, 0, 1)
        # Handle padding mismatches
        if x_out.shape[0] != x_residual.shape[0]:
            diff = x_residual.shape[0] - x_out.shape[0]
            x_out = F.pad(x_out, (0, 0, 0, 0, 0, diff)) if diff > 0 else x_out[:x_residual.shape[0]]

        return x_residual + self.dropout_layer(x_out), attn_weights


class HybridEncoder(nn.Module):
    def __init__(self, num_layers=12, embed_dim=560, attention_heads=20, alphabet="ESM-1b"):
        super().__init__()
        if not isinstance(alphabet, esm.data.Alphabet):
            alphabet = esm.data.Alphabet.from_architecture(alphabet)

        self.alphabet = alphabet
        self.pad_idx = alphabet.padding_idx
        self.embed = nn.Embedding(len(alphabet), embed_dim, padding_idx=self.pad_idx)

        self.blocks = nn.ModuleList([
            ConvUnetAttentionLayer(embed_dim, 4 * embed_dim, attention_heads)
            for _ in range(num_layers)
        ])

        # MLM Head components
        self.emb_layer_norm_after = ESM1bLayerNorm(embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=embed_dim, output_dim=len(alphabet), weight=self.embed.weight
        )

    def forward(self, tokens, need_head_weights=False):
        # Standard MLM Forward
        mask = tokens.eq(self.pad_idx)
        x = self.embed(tokens)
        if mask is not None: x = x * (1 - mask.unsqueeze(-1).type_as(x))
        x = x.transpose(0, 1)

        if not mask.any(): mask = None

        for layer in self.blocks:
            # [FIX] Pass the flag down, though ContactPredictor usually calls blocks directly
            x, _ = layer(x, mask, need_head_weights=need_head_weights)

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)
        return self.lm_head(x)  # Logits

    def get_latent(self, tokens):
        # For Stage 2 (VQ-VAE)
        mask = tokens.eq(self.pad_idx)
        x = self.embed(tokens)
        if mask is not None: x = x * (1 - mask.unsqueeze(-1).type_as(x))
        x = x.transpose(0, 1)
        if not mask.any(): mask = None

        for i, layer in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                return layer(x, mask, return_bottleneck_only=True)
            else:
                x, _ = layer(x, mask)

        return x  # Should not reach here given logic above
