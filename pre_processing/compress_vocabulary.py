import re
import json
from collections import defaultdict

# Read original vocabulary
with open('./tokens/assemble_tokens.txt', 'r') as f:
    original_tokens = [line.strip() for line in f.readlines()]

print(f"Original vocabulary size: {len(original_tokens)}")

# Check for duplicate tokens
token_counts = {}
for token in original_tokens:
    token_counts[token] = token_counts.get(token, 0) + 1

duplicate_tokens = [token for token, count in token_counts.items() if count > 1]
print(f"Number of duplicate tokens found: {len(duplicate_tokens)}")
if duplicate_tokens:
    print(f"First 5 duplicate token examples: {duplicate_tokens[:5]}")

# Create compression mapping
compressed_vocab = {}  # token text -> new ID
reverse_mapping = {}   # new ID -> token text
token_mapping = {}     # original token text -> new token text
id_mapping = {}        # original ID -> new ID
next_id = 0

# Special tokens
special_tokens = ['[PAD]', '[SEP]', '[CLS]', '[UNK]', '[MASK]']
for special in special_tokens:
    compressed_vocab[special] = next_id
    reverse_mapping[next_id] = special
    token_mapping[special] = special  # Special tokens map to themselves
    if special in original_tokens:
        original_id = original_tokens.index(special)
        id_mapping[str(original_id)] = next_id
    next_id += 1

# Process memory access patterns
mem_pattern = re.compile(r'(byte|word|dword|qword|tword|yword)\s+\[(.*?)\]')
reg_pattern = re.compile(r'r\d+|r[a-z]+|[xyz]mm\d+')
modifier_pattern = re.compile(r'\{k\d+\}|\{z\}|\{rn-sae\}')

# Memory access pattern compression
for i, token in enumerate(original_tokens):
    if token in special_tokens:
        continue  # Special tokens already processed
        
    if mem_match := mem_pattern.match(token):
        size = mem_match.group(1)
        addr = mem_match.group(2)
        
        # Replace registers with REG
        addr_pattern = re.sub(reg_pattern, 'REG', addr)
        # Replace immediates with IMM
        addr_pattern = re.sub(r'[\+\-]IMM', 'DISP', addr_pattern)
        
        # Create compressed pattern
        compressed_pattern = f"MEM_{size}[{addr_pattern}]"
        
        if compressed_pattern not in compressed_vocab:
            compressed_vocab[compressed_pattern] = next_id
            reverse_mapping[next_id] = compressed_pattern
            next_id += 1
        
        # Map original token to compressed token
        if token not in compressed_vocab:
            compressed_vocab[token] = compressed_vocab[compressed_pattern]
            # Record mapping from original token to new token
            token_mapping[token] = compressed_pattern
            # Record mapping from original ID to new ID
            id_mapping[str(i)] = compressed_vocab[compressed_pattern]

# Process registers
for i, token in enumerate(original_tokens):
    if token in special_tokens or token in token_mapping:
        continue  # Already processed tokens
        
    if re.match(r'[xyz]mm\d+', token):
        # Basic registers
        base_reg = re.match(r'([xyz]mm\d+)', token).group(1)
        reg_type = base_reg[0:3]  # xmm, ymm, zmm
        reg_num = int(base_reg[3:])
        
        # Create compressed pattern
        reg_pattern = f"{reg_type.upper()}_REG({reg_num})"
        
        if reg_pattern not in compressed_vocab:
            compressed_vocab[reg_pattern] = next_id
            reverse_mapping[next_id] = reg_pattern
            next_id += 1
        
        # Process modifiers
        modifiers = modifier_pattern.findall(token)
        for mod in modifiers:
            if mod not in compressed_vocab:
                compressed_vocab[mod] = next_id
                reverse_mapping[next_id] = mod
                next_id += 1
        
        # Map original token
        if token not in compressed_vocab:
            compressed_vocab[token] = compressed_vocab[reg_pattern]
            # Record mapping from original token to new token
            token_mapping[token] = reg_pattern
            # Record mapping from original ID to new ID
            id_mapping[str(i)] = compressed_vocab[reg_pattern]

# Process instructions and other unprocessed tokens
for i, token in enumerate(original_tokens):
    if token in special_tokens or token in token_mapping:
        continue  # Already processed tokens
        
    # May be instruction or other unprocessed token
    if token not in compressed_vocab:
        compressed_vocab[token] = next_id
        reverse_mapping[next_id] = token
        next_id += 1
        # Record mapping from original token to new token
        token_mapping[token] = token  # Keep unchanged
        # Record mapping from original ID to new ID
        id_mapping[str(i)] = next_id - 1

print(f"Compressed vocabulary size: {next_id}")
print(f"Number of mappings from original token to new token: {len(token_mapping)}")
print(f"Number of mappings from original ID to new ID: {len(id_mapping)}")

# Save compressed vocabulary and mappings
with open('./tokens/compressed_tokens.txt', 'w') as f:
    for token, idx in sorted(compressed_vocab.items(), key=lambda x: x[1]):
        if token in reverse_mapping.values():  # Only save pattern tokens
            f.write(f"{token}\n")

# Save mapping from original token to new token
with open('./tokens/token_mapping.json', 'w') as f:
    json.dump(token_mapping, f, indent=2)

# Save mapping from original ID to new ID
with open('./tokens/id_mapping.json', 'w') as f:
    json.dump(id_mapping, f, indent=2)

# Check unmapped tokens
unmapped_tokens = [token for token in original_tokens if token not in token_mapping]
if unmapped_tokens:
    print(f"Number of unmapped tokens: {len(unmapped_tokens)}")
    with open('./tokens/unmapped_tokens.txt', 'w') as f:
        for token in unmapped_tokens:
            f.write(f"{token}\n")