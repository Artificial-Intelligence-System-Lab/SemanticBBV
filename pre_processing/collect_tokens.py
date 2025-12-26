#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import re
from time import time
import sys

import angr

# iced-x86 related modules
from iced_x86 import (
    Decoder,
    DecoderOptions,
    Instruction,
    InstructionInfoFactory,
    Formatter,
    FormatterSyntax,
    FormatMnemonicOptions,
    RflagsBits,
    OpKind,
    OpAccess,
)

# If you need to use Capstone in debug mode, you can import when needed
# from capstone import *
# from capstone.x86 import *


def normalize_insn(asm):
    """
    Modified normalize_insn function:
      - Keep numbers less than 128 without replacement. Numbers can be hexadecimal or decimal.
      - For numbers greater than or equal to 128, return "PTR_ADDR", "REL_ADDR", or "IMM" based on context.
    """
    # Separate mnemonic (opcode) and operand string
    print(asm)
    opcode, op_str = asm.split("\t")

    # Remove extra spaces from operand string
    op_str = op_str.replace(" + ", "+")
    op_str = op_str.replace(" - ", "-")
    op_str = op_str.replace(" * ", "*")
    op_str = op_str.replace(" : ", ":")

    if op_str:
        opnd_strs = [x.strip() for x in op_str.split(",")]
    else:
        opnd_strs = []
    
    def repl(match):
        matched_str = match.group(0)
        try:
            # Determine if number is hexadecimal or decimal and convert to integer
            if matched_str.lower().startswith("0x"):
                # Avoid cases like "0x90h", remove trailing "h"
                s = matched_str
                if s.lower().endswith("h"):
                    s = s[:-1]
                val = int(s, 16)
            elif matched_str.lower().endswith("h"):
                # Hexadecimal numbers like "80h" (note "h" is not part of the number)
                val = int(matched_str[:-1], 16)
            else:
                # Otherwise parse as decimal
                val = int(matched_str, 10)
        except Exception:
            # If numerical parsing fails, return as is
            return matched_str

        # If number is less than 128, do not replace
        if val < 128:
            return matched_str

        # Determine special cases based on the context of the first 15 characters before the number
        start = match.start()
        preceding = op_str[max(0, start-15):start].lower()
        if "ptr" in preceding:
            return "PTR_ADDR"
        elif "rel" in preceding:
            return "REL_ADDR"
        else:
            return "IMM"
    
    # Define regex pattern to match hexadecimal numbers (0x... or ...h) and decimal numbers
    pattern = r"0x[0-9a-fA-F]+h?|\b[0-9a-fA-F]+h\b|\b[0-9]\b|(?<=[+\-*/])\d+"
    # Replace numerical values in operand string
    op_str = re.sub(pattern, repl, op_str)
    # Also replace in each operand string separated by commas
    opnd_strs = [re.sub(pattern, repl, opnd) for opnd in opnd_strs]

    return opcode, opnd_strs


def get_eflag_bit_name(bit):
    """
    Map the RflagsBits.* values in iced-x86 to descriptive strings.
    """
    if bit == RflagsBits.OF:
        return "OF"
    elif bit == RflagsBits.SF:
        return "SF"
    elif bit == RflagsBits.ZF:
        return "ZF"
    elif bit == RflagsBits.AF:
        return "AF"
    elif bit == RflagsBits.PF:
        return "PF"
    elif bit == RflagsBits.CF:
        return "CF"
    elif bit == RflagsBits.DF:
        return "DF"
    elif bit == RflagsBits.IF:
        return "IF"
    elif bit == RflagsBits.AC:
        return "AC"
    elif bit == RflagsBits.UIF:
        return "UIF"
    elif bit == RflagsBits.C0:
        return "C0"
    elif bit == RflagsBits.C1:
        return "C1"
    elif bit == RflagsBits.C2:
        return "C2"
    elif bit == RflagsBits.C3:
        return "C3"
    else:
        return "NULL"


def get_eflags_string(instruction: Instruction) -> str:
    """
    For a given instruction, analyze its read/write situation on eflags, and return a comma-separated flag string.
    """
    if not (instruction.rflags_read or instruction.rflags_modified):
        return "NULL"

    updated_flags = []

    rflags_group = [
        (instruction.rflags_read, "READ"),
        (instruction.rflags_written, "WRITTEN"),
        (instruction.rflags_cleared, "CLEARED"),
        (instruction.rflags_set, "SET"),
        (instruction.rflags_undefined, "UNDEFINED"),
    ]

    for bits, class_name in rflags_group:
        for test_bit in [
            RflagsBits.OF,
            RflagsBits.SF,
            RflagsBits.ZF,
            RflagsBits.AF,
            RflagsBits.PF,
            RflagsBits.CF,
            RflagsBits.DF,
            RflagsBits.IF,
            RflagsBits.AC,
            RflagsBits.UIF,
            RflagsBits.C0,
            RflagsBits.C1,
            RflagsBits.C2,
            RflagsBits.C3,
        ]:
            if bits & test_bit:
                name = get_eflag_bit_name(test_bit)
                if name != "NULL":
                    updated_flags.append(f"{class_name}_{name}")

    if not updated_flags:
        return "NULL"

    return ",".join(updated_flags)


def parse_operands(instruction: Instruction, info_factory: InstructionInfoFactory, debug: bool = False):
    """
    Parse iced-x86 instruction operands, return structure similar to IDA:
        [
          [op_type, op_id, is_read, is_write],
          ...
        ]
    """
    results = []
    instruction_info = info_factory.info(instruction)

    for op_index in range(instruction.op_count):
        op_kind = instruction.op_kind(op_index)
        access = instruction_info.op_access(op_index)
        is_read = 1 if access in [OpAccess.READ, OpAccess.READ_WRITE, OpAccess.READ_COND_WRITE] else 0
        is_write = 1 if access in [OpAccess.WRITE, OpAccess.READ_WRITE, OpAccess.READ_COND_WRITE] else 0

        if op_kind == OpKind.REGISTER:
            op_id = instruction.op_register(op_index)
        else:
            op_id = 0

        if debug:
            print(f"Operand[{op_index}] kind={op_kind}, id={op_id}, read={is_read}, write={is_write}")

        results.append([op_kind, op_id, is_read, is_write])

    return results


def load_token_sets(output_dir):
    """
    Load existing token sets from specified path, supports loading set format and dict format
    """
    # Define file paths to load
    assemble_tokens_path = os.path.join(output_dir, "assemble_tokens.pkl")
    mnemonic_tokens_path = os.path.join(output_dir, "mnemonic_tokens.pkl")
    op_kind_tokens_path = os.path.join(output_dir, "op_kind_tokens.pkl")
    op_id_tokens_path = os.path.join(output_dir, "op_id_tokens.pkl")
    eflags_tokens_path = os.path.join(output_dir, "eflags_tokens.pkl")
    
    # Define dict format file paths
    assemble_tokens_dict_path = os.path.join(output_dir, "assemble_tokens_dict.pkl")
    mnemonic_tokens_dict_path = os.path.join(output_dir, "mnemonic_tokens_dict.pkl")
    op_kind_tokens_dict_path = os.path.join(output_dir, "op_kind_tokens_dict.pkl")
    op_id_tokens_dict_path = os.path.join(output_dir, "op_id_tokens_dict.pkl")
    eflags_tokens_dict_path = os.path.join(output_dir, "eflags_tokens_dict.pkl")
    
    # Initialize token dictionaries
    assemble_tokens = {}
    mnemonic_tokens = {}
    op_kind_tokens = {}
    op_id_tokens = {}
    eflags_tokens = {}
    
    # Try to load existing token sets
    try:
        # Prefer to load dict format first
        if os.path.exists(assemble_tokens_dict_path):
            with open(assemble_tokens_dict_path, "rb") as f:
                assemble_tokens = pickle.load(f)
            print(f"Loaded assembly tokens dictionary: {len(assemble_tokens)} entries")
        elif os.path.exists(assemble_tokens_path):
            # If dict format does not exist, try to load set format and convert to dict
            with open(assemble_tokens_path, "rb") as f:
                token_set = pickle.load(f)
                assemble_tokens = {token: 0 for token in token_set}  # Initial frequency is 0
            print(f"Loaded assembly tokens set and converted to dictionary: {len(assemble_tokens)} entries")
        
        if os.path.exists(mnemonic_tokens_dict_path):
            with open(mnemonic_tokens_dict_path, "rb") as f:
                mnemonic_tokens = pickle.load(f)
            print(f"Loaded mnemonic tokens dictionary: {len(mnemonic_tokens)} entries")
        elif os.path.exists(mnemonic_tokens_path):
            with open(mnemonic_tokens_path, "rb") as f:
                token_set = pickle.load(f)
                mnemonic_tokens = {token: 0 for token in token_set}
            print(f"Loaded mnemonic tokens set and converted to dictionary: {len(mnemonic_tokens)} entries")
        
        if os.path.exists(op_kind_tokens_dict_path):
            with open(op_kind_tokens_dict_path, "rb") as f:
                op_kind_tokens = pickle.load(f)
            print(f"Loaded operand type tokens dictionary: {len(op_kind_tokens)} entries")
        elif os.path.exists(op_kind_tokens_path):
            with open(op_kind_tokens_path, "rb") as f:
                token_set = pickle.load(f)
                op_kind_tokens = {token: 0 for token in token_set}
            print(f"Loaded operand type tokens set and converted to dictionary: {len(op_kind_tokens)} entries")
        
        if os.path.exists(op_id_tokens_dict_path):
            with open(op_id_tokens_dict_path, "rb") as f:
                op_id_tokens = pickle.load(f)
            print(f"Loaded operand ID tokens dictionary: {len(op_id_tokens)} entries")
        elif os.path.exists(op_id_tokens_path):
            with open(op_id_tokens_path, "rb") as f:
                token_set = pickle.load(f)
                op_id_tokens = {token: 0 for token in token_set}
            print(f"Loaded operand ID tokens set and converted to dictionary: {len(op_id_tokens)} entries")
        
        if os.path.exists(eflags_tokens_dict_path):
            with open(eflags_tokens_dict_path, "rb") as f:
                eflags_tokens = pickle.load(f)
            print(f"Loaded eflags tokens dictionary: {len(eflags_tokens)} entries")
        elif os.path.exists(eflags_tokens_path):
            with open(eflags_tokens_path, "rb") as f:
                token_set = pickle.load(f)
                eflags_tokens = {token: 0 for token in token_set}
            print(f"Loaded eflags tokens set and converted to dictionary: {len(eflags_tokens)} entries")
    
    except Exception as e:
        print(f"Error loading token sets: {e}")
    
    return assemble_tokens, mnemonic_tokens, op_kind_tokens, op_id_tokens, eflags_tokens


def save_token_sets(output_dir, assemble_tokens, mnemonic_tokens, op_kind_tokens, op_id_tokens, eflags_tokens):
    """
    Save token dictionary to specified path, simultaneously save set format and dict format
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dict format token set
    with open(os.path.join(output_dir, "assemble_tokens_dict.pkl"), "wb") as f:
        pickle.dump(assemble_tokens, f)
    
    with open(os.path.join(output_dir, "mnemonic_tokens_dict.pkl"), "wb") as f:
        pickle.dump(mnemonic_tokens, f)
    
    with open(os.path.join(output_dir, "op_kind_tokens_dict.pkl"), "wb") as f:
        pickle.dump(op_kind_tokens, f)
    
    with open(os.path.join(output_dir, "op_id_tokens_dict.pkl"), "wb") as f:
        pickle.dump(op_id_tokens, f)
    
    with open(os.path.join(output_dir, "eflags_tokens_dict.pkl"), "wb") as f:
        pickle.dump(eflags_tokens, f)
    
    # Also save set format token set (backward compatible)
    with open(os.path.join(output_dir, "assemble_tokens.pkl"), "wb") as f:
        pickle.dump(set(assemble_tokens.keys()), f)
    
    with open(os.path.join(output_dir, "mnemonic_tokens.pkl"), "wb") as f:
        pickle.dump(set(mnemonic_tokens.keys()), f)
    
    with open(os.path.join(output_dir, "op_kind_tokens.pkl"), "wb") as f:
        pickle.dump(set(op_kind_tokens.keys()), f)
    
    with open(os.path.join(output_dir, "op_id_tokens.pkl"), "wb") as f:
        pickle.dump(set(op_id_tokens.keys()), f)
    
    with open(os.path.join(output_dir, "eflags_tokens.pkl"), "wb") as f:
        pickle.dump(set(eflags_tokens.keys()), f)
    
    print(f"Assembly tokens dictionary: {len(assemble_tokens)} items")
    print(f"Mnemonic tokens dictionary: {len(mnemonic_tokens)} items")
    print(f"Operand type tokens dictionary: {len(op_kind_tokens)} items")
    print(f"Operand ID tokens dictionary: {len(op_id_tokens)} items")
    print(f"Eflags tokens dictionary: {len(eflags_tokens)} items")


def main(binary_path: str, output_dir: str, debug: bool = True) -> None:
    """
    1) Traverse all saved_index.pkl files in the directory;
    2) Traverse all functions in each file, extract binary instruction data;
    3) Use iced-x86 to decode instructions, parse eflags and operand information, construct tokens;
    4) Save tokenlist to pickle file.
    """
    assemble_tokens, mnemonic_tokens, op_kind_tokens, op_id_tokens, eflags_tokens = load_token_sets(output_dir)

    info_factory = InstructionInfoFactory()
    formatter = Formatter(FormatterSyntax.NASM)
    formatter.gas_show_mnemonic_size_suffix = True
    instruction_count = 0
    processed_files = 0  # Initialize processed file counter
    
    for root, _, files in os.walk(binary_path):
        for file in files:
            if file == "saved_index.pkl":
                file_path = os.path.join(root, file)
                try:
                    # Load pickle file
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Traverse all functions
                    for func_name, func_data_list in data.items():
                        for func_data in func_data_list:
                            # Check if binary data exists
                            if len(func_data) > 2 and isinstance(func_data[2], bytes):
                                ip = func_data[0]
                                binary_data = func_data[2]
                                
                                # Use iced_x86 to decode binary data
                                decoder = Decoder(64, binary_data, ip=ip, options=DecoderOptions.NONE)
                                offset = 0
                                
                                # Decode all instructions
                                while offset < len(binary_data):
                                    old_position = decoder.position  # Record position before decoding
                                    instruction = decoder.decode()
                                    
                                    # Check if decoder advanced, if not, break loop
                                    if decoder.position == old_position:
                                        print(f"Warning: Decoder at position {offset} did not advance, might encounter invalid instruction, break loop")
                                        break
                                    
                                    ins_addr = instruction.ip
                                    mnemonic = instruction.mnemonic
                                    mnemonic_str = formatter.format_mnemonic(instruction, FormatMnemonicOptions.NO_PREFIXES)
                                    operands_str = formatter.format_all_operands(instruction)
                                    disasm = f"{mnemonic_str}\t{operands_str}"

                                    eflags = get_eflags_string(instruction)
                                    op_info = parse_operands(instruction, info_factory, debug=debug)
                                    op_code, opnd_strs = normalize_insn(disasm)

                                    # Update counter in token dictionary
                                    # For assemble_tokens
                                    assemble_tokens[op_code] = assemble_tokens.get(op_code, 0) + 1
                                    for opnd_str in opnd_strs:
                                        assemble_tokens[opnd_str] = assemble_tokens.get(opnd_str, 0) + 1
                                    
                                    # For mnemonic_tokens
                                    mnemonic_tokens[mnemonic] = mnemonic_tokens.get(mnemonic, 0) + 1
                                    
                                    # For op_kind_tokens and op_id_tokens
                                    for op in op_info:
                                        op_kind_tokens[op[0]] = op_kind_tokens.get(op[0], 0) + 1
                                        op_id_tokens[op[1]] = op_id_tokens.get(op[1], 0) + 1
                                    
                                    # For eflags_tokens
                                    eflags_tokens[eflags] = eflags_tokens.get(eflags, 0) + 1

                                    instruction_count += 1
                                    # Move to next instruction
                                    offset = decoder.position
                    
                    processed_files += 1
                    print(f"Processing file: {file_path} (processed: {processed_files} files, current instruction count: {instruction_count})")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # Save tokens to pickle file
    os.makedirs(output_dir, exist_ok=True)
    save_token_sets(output_dir, assemble_tokens, mnemonic_tokens, op_kind_tokens, op_id_tokens, eflags_tokens)
    print(f"Token set saved to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nUsage: python rewrite_angr_icedx86.py <path_to_binary> <output_dir>\n")
        sys.exit(1)

    bin_path = sys.argv[1]
    out_dir = sys.argv[2]
    # Set debug True/False as needed
    main(bin_path, out_dir, debug=False)