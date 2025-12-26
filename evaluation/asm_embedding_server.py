#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import socketserver
import struct
import pickle
import threading
import traceback

import torch
import numpy as np

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pre_processing.tokenizer import tokenize_binary_instruction
from asm_dataset_preprocessed_fine import AsmVocab
from train_rwkv7_deepspeed import RWKV7Model

# --- Reuse previous BasicBlockEncoder definition （omit comments to ensure consistency with train script) ---
class BasicBlockEncoder(torch.nn.Module):
    def __init__(self, pretrained_model, args, encoding_dim=64):
        super().__init__()
        self.num_embedding_types = pretrained_model.num_embedding_types
        self.asm_embd = pretrained_model.asm_embd
        self.mne_embd = pretrained_model.mne_embd
        self.type_embd = pretrained_model.type_embd
        self.reg_embd = pretrained_model.reg_embd
        self.rw_embd = pretrained_model.rw_embd
        self.eflag_embd = pretrained_model.eflag_embd
        self.embedding_projection_layer = pretrained_model.embedding_projection_layer
        self.model = pretrained_model.model
        self.encoding_head = torch.nn.Sequential(
            torch.nn.Linear(args.n_embd, encoding_dim),
            torch.nn.Dropout(0.1),
        )
        self.attention_query = torch.nn.Sequential(
            torch.nn.Linear(args.n_embd, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 1)
        )
        self.encoding_dim = encoding_dim

    def _get_sequence_info(self, sample_dict):
        asm_tokens = sample_dict['asm']
        batch_size, seq_len = asm_tokens.shape
        indices = torch.arange(seq_len, device=asm_tokens.device).expand(batch_size, seq_len)
        non_zero_mask = (asm_tokens != 0)
        masked_indices = torch.where(non_zero_mask, indices, torch.tensor(-1, device=asm_tokens.device))
        last_non_zero_pos = masked_indices.max(dim=1)[0]
        seq_lengths = torch.clamp(last_non_zero_pos + 1, min=1)
        return seq_lengths

    def pooling(self, x, seq_lengths):
        if x.dtype == torch.bfloat16:
            x = x.float()
        batch_size, seq_len, hidden_dim = x.shape
        device = x.device
        mask = torch.arange(seq_len, device=device).unsqueeze(0) < seq_lengths.unsqueeze(1)
        attention_scores = self.attention_query(x).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        pooled = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
        return pooled

    def _encode_single_sample(self, sample_dict):
        asm_emb   = self.asm_embd(sample_dict['asm'])
        mne_emb   = self.mne_embd(sample_dict['mne'])
        type_emb  = self.type_embd(sample_dict['type'])
        reg_emb   = self.reg_embd(sample_dict['reg'])
        rw_emb    = self.rw_embd(sample_dict['rw'])
        eflag_emb = self.eflag_embd(sample_dict['eflag'])

        seq_lengths = self._get_sequence_info(sample_dict)
        combined_emb = torch.cat((asm_emb, mne_emb, type_emb, reg_emb, rw_emb, eflag_emb), dim=-1)
        projected_emb = self.embedding_projection_layer(combined_emb)
        hidden_states = self.model(projected_emb)
        pooled_output = self.pooling(hidden_states, seq_lengths)
        encoding = self.encoding_head(pooled_output)
        return encoding

    def encode(self, sample_dict):
        with torch.no_grad():
            return self._encode_single_sample(sample_dict)

# --- Load model and vocabulary ---
def load_model_from_checkpoint(checkpoint_path, vocabs_dir, device,
                               n_embd, n_layer, head_size, encoding_dim):
    print(f"[INIT] Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Load various vocabulary
    vocab_files = {
        "asm":   "asm_tokens.txt",
        "mne":   "mne_tokens.txt",
        "type":  "type_tokens.txt",
        "reg":   "reg_tokens.txt",
        "rw":    "rw_tokens.txt",
        "eflag": "eflag_tokens.txt"
    }
    vocabs = {}
    for key, fn in vocab_files.items():
        path = os.path.join(vocabs_dir, fn)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Vocab file missing: {path}")
        v = AsmVocab(); v.load(path); vocabs[key] = v

    sizes = {k: v.length() for k, v in vocabs.items()}
    class Args: pass
    args = Args()
    args.n_embd   = n_embd
    args.n_layer  = n_layer
    args.head_size= head_size

    pretrained = RWKV7Model(args, sizes)
    model = BasicBlockEncoder(pretrained, args, encoding_dim)

    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("Using BF16 mixed precision")
        model.asm_embd = model.asm_embd.to(torch.bfloat16)
        model.mne_embd = model.mne_embd.to(torch.bfloat16)
        model.type_embd = model.type_embd.to(torch.bfloat16)
        model.reg_embd = model.reg_embd.to(torch.bfloat16)
        model.rw_embd = model.rw_embd.to(torch.bfloat16)
        model.eflag_embd = model.eflag_embd.to(torch.bfloat16)
        model.embedding_projection_layer = model.embedding_projection_layer.to(torch.bfloat16)
        model.model = model.model.to(torch.bfloat16)

    model.eval()
    model.to(device)
    print(f"[INIT] Model ready. encoding_dim={encoding_dim}")
    return model, vocabs

# --- Build single sample input ---
def prepare_sample_for_model(tokens, vocabs):
    """
    Convert tokenized basic block to model input:
    - Ensure all feature lists have consistent length
    - pad to multiple of 16
    """
    import torch

    # Initialize empty list
    sample = {k: [] for k in ['asm','mne','type','reg','rw','eflag']}

    # Iterate through each token and extend to corresponding feature sequence
    for tok in tokens:
        # asm: multiple opcode tokens
        sample['asm'].extend([vocabs['asm'].get_id(x) for x in tok['asm']])
        # mne: one mnemonic token, repeated len(tok['asm']) times
        sample['mne'].extend([vocabs['mne'].get_id(tok['mne'])] * len(tok['asm']))
        # type: list with each (token, count) pair
        for t, cnt in tok['type']:
            sample['type'].extend([vocabs['type'].get_id(t)] * cnt)
        # reg: possibly multiple register tokens
        sample['reg'].extend([vocabs['reg'].get_id(r) for r in tok['reg']])
        # rw: list with each (token, count) pair
        for r, cnt in tok['rw']:
            sample['rw'].extend([vocabs['rw'].get_id(r)] * cnt)
        # eflag: one flag token, repeated len(tok['asm']) times
        sample['eflag'].extend([vocabs['eflag'].get_id(tok['eflag'])] * len(tok['asm']))

    # Verify all feature lengths are consistent
    seq_len = len(sample['asm'])
    for k, lst in sample.items():
        if len(lst) != seq_len:
            raise ValueError(f"Feature {k} length {len(lst)} != asm length {seq_len}")

    # Calculate padded length: round up to multiple of 16
    pad_to = ((seq_len + 15) // 16) * 16
    pad_len = pad_to - seq_len

    # Pad all features
    if pad_len > 0:
        for k in sample:
            sample[k].extend([0] * pad_len)

    # Convert to tensor and add batch dimension
    for k in sample:
        sample[k] = torch.tensor(sample[k], dtype=torch.long).unsqueeze(0)

    return sample

# --- TCP Server Handler ---
class InferenceHandler(socketserver.BaseRequestHandler):
    def handle(self):
        addr = self.client_address[0]
        print(f"[{addr}] Connected")
        while True:
            hdr = self._recvall(4)
            if not hdr:
                break
            (n_bytes,) = struct.unpack('!I', hdr)
            if n_bytes == 0:
                print(f"[{addr}] Shutdown request")
                break
            data = self._recvall(n_bytes)
            if data is None:
                break

            try:
                # 1) tokenize
                toks = tokenize_binary_instruction(data, ip=0)
                # 2) prepare sample
                sample = prepare_sample_for_model(
                    toks,
                    self.server.vocabs,
                )
                # 3) move to device
                for k in sample: sample[k] = sample[k].to(self.server.device)
                # 4) infer
                with torch.no_grad():
                    emb = self.server.model.encode(sample)  # [1, D]
                vec = emb[0].cpu().numpy().astype(np.float32)
                # 5) reply
                payload = vec.tobytes()  # Directly convert float32 array to byte stream
                out_hdr = struct.pack('!I', len(payload))
                self.request.sendall(out_hdr + payload)
            except Exception as e:
                print(f"[{addr}] Error during inference: {e}")
                traceback.print_exc()
                break
        print(f"[{addr}] Disconnected")

    def _recvall(self, n):
        buf = bytearray(n); mv = memoryview(buf)
        total = 0
        while total < n:
            cnt = self.request.recv_into(mv[total:], n - total)
            if cnt == 0:
                return None
            total += cnt
        return bytes(buf)

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

def serve(host, port, checkpoint, vocab_dir, device,
          n_embd, n_layer, head_size, encoding_dim):
    model, vocabs = load_model_from_checkpoint(
        checkpoint, vocab_dir, device,
        n_embd, n_layer, head_size, encoding_dim
    )
    ThreadedTCPServer.model       = model
    ThreadedTCPServer.vocabs      = vocabs
    ThreadedTCPServer.device      = device

    with ThreadedTCPServer((host, port), InferenceHandler) as server:
        print(f"[SERVER] Listening on {host}:{port}")
        server.serve_forever()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--host',       type=str, default='0.0.0.0')
    p.add_argument('--port',       type=int, default=12345)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--vocab_dir',  type=str, required=True)
    p.add_argument('--device',     type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--n_embd',     type=int, default=768)
    p.add_argument('--n_layer',    type=int, default=6)
    p.add_argument('--head_size',  type=int, default=64)
    p.add_argument('--encoding_dim', type=int, default=128)
    args = p.parse_args()
    serve(
        args.host, args.port,
        args.checkpoint, args.vocab_dir,
        args.device,
        args.n_embd, args.n_layer, args.head_size,
        args.encoding_dim
    )