import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from esm.modules import TransformerLayer, ESM1bLayerNorm
from .modules import LayerNorm1d


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_(0, 0.02)
        self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=1)

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', self.embedding.weight.data.clone())

    def forward(self, inputs, mask_1d=None):
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        inputs_norm = F.normalize(inputs, p=2, dim=2)
        embed_norm = F.normalize(self.embedding.weight, p=2, dim=1)

        if mask_1d is not None:
            valid_mask = ~mask_1d
            valid_count = torch.sum(valid_mask).float()
            flat_input = inputs_norm[valid_mask].contiguous()
        else:
            valid_count = float(inputs.size(0) * inputs.size(1))
            flat_input = inputs_norm.view(-1, self.embedding_dim)
            valid_mask = None

        distances = 2 - 2 * torch.matmul(flat_input, embed_norm.t())
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized_valid = torch.matmul(encodings, embed_norm)

        # EMA Logic (omitted for inference brevity, but kept in structure if needed)

        if valid_mask is not None:
            full_quantized = torch.zeros_like(inputs)
            flat_output = torch.zeros(inputs.shape[0] * inputs.shape[1], self.embedding_dim, device=inputs.device,
                                      dtype=inputs.dtype)
            flat_output[valid_mask.view(-1)] = quantized_valid
            full_quantized = flat_output.view(input_shape)
        else:
            full_quantized = quantized_valid.view(input_shape)

        e_loss = F.mse_loss(full_quantized.detach(), inputs_norm, reduction='sum') / (valid_count + 1e-6)
        vq_loss = self.commitment_cost * e_loss
        quantized_output = inputs_norm + (full_quantized - inputs_norm).detach()

        return vq_loss, quantized_output.permute(0, 2, 1).contiguous(), e_loss.detach(), encoding_indices


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_quantizers=1, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, decay)
            for _ in range(num_quantizers)
        ])

    def forward(self, x, mask_1d=None):
        # Simplified forward for single layer usage in your training code
        vq_loss, quantized, recon_loss, indices = self.layers[0](x, mask_1d)
        return vq_loss, quantized, recon_loss, indices


class TransformerAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=12, num_layers=4):
        super().__init__()
        self.pre_proj = nn.Linear(input_dim, output_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim=output_dim, ffn_embed_dim=output_dim * 4, attention_heads=num_heads,
                             add_bias_kv=False, use_esm1b_layer_norm=True, use_rotary_embeddings=True)
            for _ in range(num_layers)
        ])
        self.norm = ESM1bLayerNorm(output_dim)

    def forward(self, x, padding_mask=None):
        # Input: [B, C, L]
        x = x.permute(2, 0, 1)  # -> [L, B, C]
        x = self.pre_proj(x)
        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask)
        x = self.norm(x)
        return x.permute(1, 2, 0)  # -> [B, C, L]


class TransformerDecoderAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=20, num_layers=16):
        super().__init__()
        self.pre_proj = nn.Linear(input_dim, output_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim=output_dim, ffn_embed_dim=output_dim * 4, attention_heads=num_heads,
                             add_bias_kv=False, use_esm1b_layer_norm=True, use_rotary_embeddings=True)
            for _ in range(num_layers)
        ])
        self.norm = ESM1bLayerNorm(output_dim)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.pre_proj(x)
        for layer in self.layers: x, _ = layer(x)
        x = self.norm(x)
        return x.permute(1, 2, 0)


class HybridDecoder(nn.Module):
    def __init__(self, embed_dim, vocab_size, num_heads=20):
        super().__init__()
        self.global_attn_1_layers = nn.ModuleList([
            TransformerLayer(embed_dim, 4 * embed_dim, num_heads, add_bias_kv=False,
                             use_esm1b_layer_norm=True, use_rotary_embeddings=True)
            for _ in range(8)
        ])
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1),
            LayerNorm1d(embed_dim), nn.GELU()
        )
        self.global_attn_2 = nn.ModuleList([
            TransformerLayer(embed_dim, 4 * embed_dim, num_heads, add_bias_kv=False,
                             use_esm1b_layer_norm=True, use_rotary_embeddings=True)
            for _ in range(8)
        ])
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1),
            LayerNorm1d(embed_dim), nn.GELU()
        )
        self.final_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=3)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, super_tokens, mask_l4=None, mask_l2=None):
        x = super_tokens.permute(2, 0, 1)
        for layer in self.global_attn_1_layers: x, _ = layer(x, self_attn_padding_mask=mask_l4)
        x = x.permute(1, 2, 0)
        x = self.up1(x)
        x = x.permute(2, 0, 1)
        for layer in self.global_attn_2: x, _ = layer(x, self_attn_padding_mask=mask_l2)
        x = x.permute(1, 2, 0)
        x = self.up2(x)
        x = self.final_conv(x)
        x = x.permute(0, 2, 1)
        x = self.norm_out(x)
        return self.lm_head(x)


class ProteinVQAutoencoder(nn.Module):
    def __init__(self, encoder, adapter, vq_layer, decoder_adapter, decoder):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.vq = vq_layer
        self.decoder_adapter = decoder_adapter
        self.decoder = decoder
        self.input_norm = nn.LayerNorm(560)
        self.pad_idx = encoder.pad_idx

        # 冻结 Encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def downsample_mask(self, mask, scale):
        m = (~mask).float().unsqueeze(1)
        m = F.max_pool1d(m, kernel_size=scale, stride=scale, ceil_mode=False)
        return ~(m.bool().squeeze(1))

    def forward(self, tokens, vq_mix_ratio=1.0):
        B, L = tokens.shape
        remainder = L % 4
        if remainder != 0:
            pad_amt = 4 - remainder
            tokens = F.pad(tokens, (0, pad_amt), value=self.pad_idx)

        with torch.no_grad():
            encoded_output = self.encoder.get_latent(tokens)
            if encoded_output.shape[1] == B:
                encoded_output = encoded_output.permute(1, 2, 0)
            elif encoded_output.shape[0] != B:
                encoded_output = encoded_output.permute(1, 2, 0)

        normed_tokens = self.input_norm(encoded_output.permute(0, 2, 1)).permute(0, 2, 1)

        padding_mask = tokens.eq(self.pad_idx)
        mask_l2 = self.downsample_mask(padding_mask, 2)
        mask_l4 = self.downsample_mask(mask_l2, 2)

        super_tokens_trainable = self.adapter(normed_tokens, padding_mask=mask_l4)
        L_feat = super_tokens_trainable.shape[2]
        if mask_l4.shape[1] > L_feat:
            mask_l4 = mask_l4[:, :L_feat]
        elif mask_l4.shape[1] < L_feat:
            mask_l4 = F.pad(mask_l4, (0, L_feat - mask_l4.shape[1]), value=True)
        vq_loss, quantized_raw, latent_loss, encoding_indices = self.vq(super_tokens_trainable, mask_1d=mask_l4)
        decoder_input = quantized_raw
        restored_features = self.decoder_adapter(decoder_input)
        L_mid = restored_features.shape[2] * 2
        if mask_l2.shape[1] > L_mid:
            mask_l2 = mask_l2[:, :L_mid]
        elif mask_l2.shape[1] < L_mid:
            mask_l2 = F.pad(mask_l2, (0, L_mid - mask_l2.shape[1]), value=True)

        logits = self.decoder(restored_features, mask_l4, mask_l2)

        return logits, vq_loss, latent_loss, encoding_indices


class ProteinCodeDecoder(nn.Module):
    def __init__(self, vq_layer, decoder_adapter, decoder):
        super().__init__()
        self.codebook = vq_layer.layers[0].embedding
        self.decoder_adapter = decoder_adapter
        self.decoder = decoder

    def forward(self, codes):
        B, L = codes.shape
        device = codes.device
        x = self.codebook(codes)
        x = x.permute(0, 2, 1)
        restored_features = self.decoder_adapter(x)
        L_feat = restored_features.shape[2]
        mask_l4 = torch.zeros((B, L_feat), dtype=torch.bool, device=device)
        L_mid = L_feat * 2
        mask_l2 = torch.zeros((B, L_mid), dtype=torch.bool, device=device)
        logits = self.decoder(restored_features, mask_l4, mask_l2)
        aa_indices = logits.argmax(dim=1)
        return aa_indices
