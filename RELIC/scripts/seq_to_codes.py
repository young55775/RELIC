import argparse
import torch
from collections import OrderedDict
import esm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relic.vq import ProteinVQAutoencoder, TransformerAdapter, TransformerDecoderAdapter, ResidualVectorQuantizer, \
    HybridDecoder
from relic.encoder import HybridEncoder


def load_vq_model(encoder_ckpt, vq_ckpt, device, alphabet):
    # 1. Instantiate based on training config
    encoder = HybridEncoder(embed_dim=560, alphabet=alphabet)
    adapter = TransformerAdapter(560, 384, num_heads=12, num_layers=4)
    vq = ResidualVectorQuantizer(8192, 384, 1)
    # Note: Training config says 16 layers for decoder adapter
    decoder_adapter = TransformerDecoderAdapter(384, 560, num_heads=20, num_layers=16)
    decoder = HybridDecoder(560, len(alphabet), num_heads=20)

    model = ProteinVQAutoencoder(encoder, adapter, vq, decoder_adapter, decoder).to(device)

    # 2. Load Encoder
    print(f"Loading Encoder: {encoder_ckpt}")
    esm_sd = torch.load(encoder_ckpt, map_location='cpu')
    if 'model_state_dict' in esm_sd: esm_sd = esm_sd['model_state_dict']
    new_esm = OrderedDict()
    for k, v in esm_sd.items():
        n = k.replace("module.", "").replace("_orig_mod.", "").replace("base_model.", "")
        if "layers." in n: n = n.replace("layers.", "blocks.")
        if "embed_tokens" in n: n = n.replace("embed_tokens", "embed")
        new_esm[n] = v
    model.encoder.load_state_dict(new_esm, strict=False)

    # 3. Load VQ Parts
    print(f"Loading VQ: {vq_ckpt}")
    vq_sd = torch.load(vq_ckpt, map_location='cpu')  # Contains {'adapter':..., 'vq':...}

    # Map checkpoint keys to model modules
    comps = {
        'adapter': model.adapter,
        'vq': model.vq,
        'decoder_adapter': model.decoder_adapter,
        'decoder': model.decoder
    }

    for key, sub_sd in vq_sd.items():
        if key in comps:
            clean_sub = OrderedDict()
            for k, v in sub_sd.items():
                clean_sub[k.replace("module.", "").replace("_orig_mod.", "")] = v
            comps[key].load_state_dict(clean_sub, strict=True)

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--vq_ckpt", required=True)
    parser.add_argument("--seq", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")

    model = load_vq_model(args.encoder_ckpt, args.vq_ckpt, device, alphabet)

    # Tokenize
    tokens = [alphabet.cls_idx] + list(alphabet.encode(args.seq)) + [alphabet.eos_idx]
    pad_len = (4 - len(tokens) % 4) % 4
    tokens += [alphabet.padding_idx] * pad_len
    t_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        # forward returns: logits, vq_loss, latent_loss, indices
        _, _, _, indices = model(t_tensor)

    codes = indices.view(-1).cpu().tolist()
    valid_len = (len(tokens) - pad_len + 3) // 4
    print(f"\nCodes: {codes[:valid_len]}")


if __name__ == "__main__":
    main()
