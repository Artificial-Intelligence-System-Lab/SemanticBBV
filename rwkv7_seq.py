import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict

# Performance optimization related settings
torch.backends.cudnn.benchmark = True  # Optimize CUDNN for fixed size input
torch.backends.cudnn.allow_tf32 = True  # Allow use of TF32 format (on Ampere and above GPUs)
torch.backends.cuda.matmul.allow_tf32 = True  # Allow matrix multiplication to use TF32

# Disable automatic mixed precision
torch._C._jit_set_autocast_mode(False)

# TorchScript related settings
#MyModule = torch.jit.ScriptModule
#MyFunction = torch.jit.script_method
#MyStatic = torch.jit.script

MyModule = nn.Module  # Changed to ordinary Module to avoid TorchScript issues
MyFunction = lambda x: x  # Use ordinary function decorator
MyStatic = staticmethod

# Enable CUDA kernel
USE_CUDA_KERNEL = True
HEAD_SIZE = 64  # Need to set HEAD_SIZE, adjust according to your model configuration
DTYPE = torch.bfloat16  # Or use torch.float16, depending on your needs

D_DECAY_LORA = 64
D_AAA_LORA = 64
D_MV_LORA = 32
D_GATE_LORA = 128

from torch.utils.cpp_extension import load

load(name="wkv7s", sources=["cuda/wkv7s_op.cpp", f"cuda/wkv7s.cu"], is_python_module=False,
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
class WKV_7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b):
        with torch.no_grad():
            T, C = r.size()
            H = C // HEAD_SIZE
            N = HEAD_SIZE
            assert HEAD_SIZE == C // H
            assert all(x.dtype == DTYPE for x in [r,w,k,v,a,b])
            assert all(x.is_contiguous() for x in [r,w,k,v,a,b])
            y = torch.empty((T, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            torch.ops.wkv7s.forward(1, T, C, H, state, r, w, k, v, a, b, y)
            return y
def RWKV7_OP(state, r, w, k, v, a, b):
    return WKV_7.apply(state, r, w, k, v, a, b)


class RWKV_x070(MyModule):
    def __init__(self, args, model_path):
        super().__init__()
        self.args = args
        self.n_embd = args.n_embd
        self.n_layer = args.n_layer
        self.eval()
        
        model = torch.load(model_path, map_location='cuda')
        model = {key[6:] if key.startswith("model.") else key: value for key, value in model.items()}
        self.z = model
        z = self.z
        self.n_head, self.head_size = z['blocks.0.att.r_k'].shape

        keys = list(z.keys())
        for k in keys:
            if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                z[k] = z[k].t()
            z[k] = z[k].squeeze().to(dtype=DTYPE)
            if k.endswith('att.r_k'): z[k] = z[k].flatten()
        assert self.head_size == args.head_size

        # z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
        z['blocks.0.att.v0'] = z['blocks.0.att.a0'] # actually ignored
        z['blocks.0.att.v1'] = z['blocks.0.att.a1'] # actually ignored
        z['blocks.0.att.v2'] = z['blocks.0.att.a2'] # actually ignored
        
        # Set all parameters to not require gradient computation
        self.set_requires_grad_false()
    
    def set_requires_grad_false(self):
        """Set all parameters to not require gradient computation"""
        for k in self.z.keys():
            if isinstance(self.z[k], torch.Tensor):
                # For non-leaf tensors, first use detach() to separate the computation graph
                if not self.z[k].is_leaf:
                    self.z[k] = self.z[k].detach()
                self.z[k].requires_grad_(False)
        return self

    def forward(self, idx:dict, state, full_output=False, batch_size=None):
        if state == None:
            # Check if batch_size is provided
            if batch_size is None:
                # Try to infer batch_size from input
                if isinstance(idx['asm'], list):
                    batch_size = 1
                else:
                    batch_size = idx['asm'].shape[0] if len(idx['asm'].shape) > 1 else 1
            
            state = [None for _ in range(self.args.n_layer * 3)]
            for i in range(self.args.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
                state[i*3+0] = torch.zeros(batch_size, self.args.n_embd, dtype=DTYPE, requires_grad=False, device="cuda")
                state[i*3+1] = torch.zeros(batch_size, self.args.n_embd // self.args.head_size, self.args.head_size, self.args.head_size, dtype=torch.float, requires_grad=False, device="cuda")
                state[i*3+2] = torch.zeros(batch_size, self.args.n_embd, dtype=DTYPE, requires_grad=False, device="cuda")
        
        # Process batch input
        if isinstance(idx['asm'], torch.Tensor) and len(idx['asm'].shape) > 1:
            # Batch input
            return self.forward_batch(idx, state, full_output)
        elif type(idx['asm']) is list:
            if len(idx['asm']) > 1:
                return self.forward_seq(idx, state, full_output)
            else:
                return self.forward_one(idx, state)
        else:
            return self.forward_one(idx, state)

    @MyFunction
    def forward_one(self, token_inputs:dict, state:List[torch.Tensor]):
        with torch.no_grad(): 
            z = self.z

            # Process multiple embeddings
            asm_emb = z['asm_embd.weight'][token_inputs['asm']]
            mne_emb = z['mne_embd.weight'][token_inputs['mne']]
            type_emb = z['type_embd.weight'][token_inputs['type']]
            reg_emb = z['reg_embd.weight'][token_inputs['reg']]
            rw_emb = z['rw_embd.weight'][token_inputs['rw']]
            eflag_emb = z['eflag_embd.weight'][token_inputs['eflag']]
            
            combined_emb_concatenated = torch.cat(
                (asm_emb, mne_emb, type_emb, reg_emb, rw_emb, eflag_emb), 
                dim=-1
            )
            
            # Through projection layer
            x = F.linear(combined_emb_concatenated, 
                    z['embedding_projection_layer.weight'], 
                    z['embedding_projection_layer.bias'])

            # Apply LN
            x = F.layer_norm(x, (self.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_one(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx, state[i*3+2] = RWKV_x070_CMix_one(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            return x, state
        
    @MyFunction
    def forward_batch(self, token_inputs:dict, state:List[torch.Tensor], full_output:bool=False):
        """Process batch input data
        
        Args:
            token_inputs: Dictionary containing multiple embedding types, each value is a tensor of shape [B, T] or [B, T, D]
            state: Model state list
            full_output: Whether to return output for all time steps
            
        Returns:
            Output tensor and updated state
        """
        with torch.no_grad(): 
            z = self.z
            
            # Get batch size and sequence length
            batch_size = token_inputs['asm'].shape[0]
            seq_len = token_inputs['asm'].shape[1] if len(token_inputs['asm'].shape) > 1 else 1
            
            # Process multiple embeddings
            asm_emb = z['asm_embd.weight'][token_inputs['asm']]  # [B, T] -> [B, T, D]
            mne_emb = z['mne_embd.weight'][token_inputs['mne']]  # [B, T] -> [B, T, D]
            type_emb = z['type_embd.weight'][token_inputs['type']]  # [B, T] -> [B, T, D]
            reg_emb = z['reg_embd.weight'][token_inputs['reg']]  # [B, T] -> [B, T, D]
            rw_emb = z['rw_embd.weight'][token_inputs['rw']]  # [B, T] -> [B, T, D]
            eflag_emb = z['eflag_embd.weight'][token_inputs['eflag']]  # [B, T] -> [B, T, D]
            
            # Merge all embeddings [B, T, D1+D2+...]
            combined_emb_concatenated = torch.cat(
                (asm_emb, mne_emb, type_emb, reg_emb, rw_emb, eflag_emb), 
                dim=-1
            )
            
            # Through projection layer [B, T, D]
            x = F.linear(combined_emb_concatenated, 
                    z['embedding_projection_layer.weight'], 
                    z['embedding_projection_layer.bias'])

            # Reshape to [B*T, D] for processing
            x_reshaped = x.reshape(-1, self.n_embd)
            
            # Apply LN
            x_reshaped = F.layer_norm(x_reshaped, (self.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
            
            # Reshape back to [B, T, D]
            x = x_reshaped.reshape(batch_size, seq_len, self.n_embd)
            
            # Prepare state for each batch
            batch_states = []
            for b in range(batch_size):
                # Extract the state for the current batch
                current_states = []
                for i in range(self.args.n_layer * 3):
                    if i % 3 == 1:  # att_kv state
                        current_states.append(state[i][b].unsqueeze(0))  # [1, H, N, N]
                    else:  # att_x_prev and ffn_x_prev states
                        current_states.append(state[i][b].unsqueeze(0))  # [1, D]
                batch_states.append(current_states)
            
            # Process sequences separately for each batch
            outputs = []
            new_states = [[] for _ in range(self.args.n_layer * 3)]
            
            for b in range(batch_size):
                # Get the input sequence for the current batch
                x_b = x[b]  # [T, D]
                
                # Initialize v_first
                v_first = torch.empty_like(x_b)
                
                # Get the state for the current batch
                current_state = batch_states[b]
                
                # Process each layer
                for i in range(self.n_layer):
                    bbb = f'blocks.{i}.'
                    att = f'blocks.{i}.att.'
                    ffn = f'blocks.{i}.ffn.'

                    xx = F.layer_norm(x_b, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                    xx, current_state[i*3+0], current_state[i*3+1], v_first = RWKV_x070_TMix_seq(
                        i, self.n_head, self.head_size, xx, current_state[i*3+0], v_first, current_state[i*3+1],
                        z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                        z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], 
                        z[att+'v0'], z[att+'v1'], z[att+'v2'], z[att+'g1'], z[att+'g2'], z[att+'k_k'], 
                        z[att+'k_a'], z[att+'r_k'], z[att+'receptance.weight'], z[att+'key.weight'], 
                        z[att+'value.weight'], z[att+'output.weight'], z[att+'ln_x.weight'], z[att+'ln_x.bias']
                    )
                    x_b = x_b + xx

                    xx = F.layer_norm(x_b, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                    xx, current_state[i*3+2] = RWKV_x070_CMix_seq(
                        xx, current_state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight']
                    )
                    x_b = x_b + xx
                
                # Collect the updated states
                for i in range(len(current_state)):
                    new_states[i].append(current_state[i])
                
                # Decide whether to keep only the last time step based on full_output
                if not full_output: 
                    x_b = x_b[-1,:]
                
                # Apply final layer normalization
                if not full_output:
                    x_b = F.layer_norm(x_b, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
                else:
                    x_b_reshaped = x_b.reshape(-1, self.n_embd)
                    x_b_reshaped = F.layer_norm(x_b_reshaped, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
                    x_b = x_b_reshaped.reshape(seq_len, self.n_embd)
                
                outputs.append(x_b)
            
            # Merge outputs from all batches
            if full_output:
                # If keeping all time steps, output shape is [B, T, D]
                output = torch.stack(outputs, dim=0)
            else:
                # If only keeping the last time step, output shape is [B, D]
                output = torch.stack(outputs, dim=0)
            
            # Merge states from all batches
            for i in range(len(new_states)):
                if i % 3 == 1:  # att_kv state
                    state[i] = torch.cat(new_states[i], dim=0)  # [B, H, N, N]
                else:  # att_x_prev and ffn_x_prev states
                    state[i] = torch.cat(new_states[i], dim=0)  # [B, D]
            
            return output, state

    @MyFunction
    def forward_seq(self, token_inputs:dict, state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            z = self.z
            # Process multiple embeddings
            asm_emb = z['asm_embd.weight'][token_inputs['asm']]
            mne_emb = z['mne_embd.weight'][token_inputs['mne']]
            type_emb = z['type_embd.weight'][token_inputs['type']]
            reg_emb = z['reg_embd.weight'][token_inputs['reg']]
            rw_emb = z['rw_embd.weight'][token_inputs['rw']]
            eflag_emb = z['eflag_embd.weight'][token_inputs['eflag']]
            
            combined_emb_concatenated = torch.cat(
                (asm_emb, mne_emb, type_emb, reg_emb, rw_emb, eflag_emb), 
                dim=-1
            )
            
            # Through projection layer
            x = F.linear(combined_emb_concatenated, 
                    z['embedding_projection_layer.weight'], 
                    z['embedding_projection_layer.bias'])

            # Apply LN
            x = F.layer_norm(x, (self.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_seq(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx, state[i*3+2] = RWKV_x070_CMix_seq(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            if not full_output: x = x[-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            return x, state

########################################################################################################

@MyStatic
def RWKV_x070_TMix_one(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    xx = x_prev - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
    w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)

    vk = v.view(H,N,1) @ k.view(H,1,N)
    ab = (-kk).view(H,N,1) @ (kk*a).view(H,1,N)
    state = state * w.view(H,1,N) + state @ ab.float() + vk.float()
    xx = (state.to(dtype=x.dtype) @ r.view(H,N,1))

    xx = torch.nn.functional.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)    
    xx = xx + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    return (xx * g) @ O_, x, state, v_first

@MyStatic
def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    T = x.shape[0]
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    ######## cuda-free method 
    # w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)
    # for t in range(T):
    #     r_, w_, k_, v_, kk_, a_ = r[t], w[t], k[t], v[t], kk[t], a[t]
    #     vk = v_.view(H,N,1) @ k_.view(H,1,N)
    #     ab = (-kk_).view(H,N,1) @ (kk_*a_).view(H,1,N)
    #     state = state * w_.view(H,1,N) + state @ ab.float() + vk.float()
    #     xx[t] = (state.to(dtype=x.dtype) @ r_.view(H,N,1)).view(H*N)

    w = -torch.nn.functional.softplus(-(w0 + w)) - 0.5
    xx = RWKV7_OP(state, r, w, k, v, -kk, kk*a)

    xx = torch.nn.functional.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
    xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
    return (xx * g) @ O_, x[-1,:], state, v_first

########################################################################################################

@MyStatic
def RWKV_x070_CMix_one(x, x_prev, x_k, K_, V_):
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_, x

@MyStatic
def RWKV_x070_CMix_seq(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_, x[-1,:]
