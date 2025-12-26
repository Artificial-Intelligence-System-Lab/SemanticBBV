#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from types import ModuleType
from typing import Dict
from iced_x86 import (
    Decoder,
    DecoderOptions,
    Instruction,
    Formatter,
    FormatterSyntax,
    FormatMnemonicOptions,
    InstructionInfoFactory,
    RflagsBits,
    OpAccess,
    OpCodeOperandKind,
    Register,
    RegisterInfo,
    Code,
    Mnemonic
)
import ipdb
import sys

def create_enum_dict(module: ModuleType) -> Dict[int, str]:
    return {module.__dict__[key]:key for key in module.__dict__ if isinstance(module.__dict__[key], int)}

OP_ACCESS_TO_STRING: Dict[int, str] = create_enum_dict(OpAccess)

def op_access_to_string(value: int) -> str:
    """
    Convert the enumeration value of operation access type to the corresponding string representation
    
    Args:
        value: OpAccess enumeration value
        
    Returns:
        str: Name corresponding to the enumeration value, or string representation of the value if not found
    """
    s = OP_ACCESS_TO_STRING.get(value)
    if s is None:
        return str(value)
    return s

OP_CODE_OPERAND_KIND_TO_STRING: Dict[int, str] = create_enum_dict(OpCodeOperandKind)
def op_code_operand_kind_to_string(value: int) -> str:
    s = OP_CODE_OPERAND_KIND_TO_STRING.get(value)
    if s is None:
        return str(value)
    return s

REGISTER_TO_STRING: Dict[int, str] = create_enum_dict(Register)
STRING_TO_REGISTER: Dict[str, int] = {value:key for key, value in REGISTER_TO_STRING.items()}

# Manually add GPR8 register mapping to corresponding 16-bit registers
gpr8_to_gpr16_mapping = {
    # Classic 8-bit register to 16-bit register mapping
    "AL": Register.AX,  # AL is the low byte of AX
    "BL": Register.BX,  # BL is the low byte of BX
    "CL": Register.CX,  # CL is the low byte of CX
    "DL": Register.DX,  # DL is the low byte of DX
    "AH": Register.AX,  # AH is the high byte of AX
    "BH": Register.BX,  # BH is the high byte of BX
    "CH": Register.CX,  # CH is the high byte of CX
    "DH": Register.DX,  # DH is the high byte of DX
    
    # Extended 8-bit register to 16-bit register mapping
    "DIL": Register.DI,  # DIL is the low byte of DI
    "SIL": Register.SI,  # SIL is the low byte of SI
    "BPL": Register.BP,  # BPL is the low byte of BP
    "SPL": Register.SP,  # SPL is the low byte of SP
    
    # 8-bit registers added in 64-bit architecture mapped to 16-bit registers
    "R8B": Register.R8W,  # R8B is the low byte of R8W
    "R9B": Register.R9W,  # R9B is the low byte of R9W
    "R10B": Register.R10W,  # R10B is the low byte of R10W
    "R11B": Register.R11W,  # R11B is the low byte of R11W
    "R12B": Register.R12W,  # R12B is the low byte of R12W
    "R13B": Register.R13W,  # R13B is the low byte of R13W
    "R14B": Register.R14W,  # R14B is the low byte of R14W
    "R15B": Register.R15W   # R15B is the low byte of R15W
}

# Add GPR8 registers to STRING_TO_REGISTER dictionary, mapped to corresponding 16-bit registers
STRING_TO_REGISTER.update(gpr8_to_gpr16_mapping)

def register_to_string(value: int) -> str:
    s = REGISTER_TO_STRING.get(value)
    if s is None:
        return str(value)
    return s

CODE_TO_STRING: Dict[int, str] = create_enum_dict(Code)
def code_to_string(value: int) -> str:
    s = CODE_TO_STRING.get(value)
    if s is None:
        return str(value)
    return s

MNEMONIC_TO_STRING: Dict[int, str] = create_enum_dict(Mnemonic)
def mnemonic_to_string(value: int) -> str:
    s = MNEMONIC_TO_STRING.get(value)
    if s is None:
        return str(value)
    return s

# Precompiled regular expression patterns
# Supported tokens: registers, numbers, operators, punctuation, brackets, prefixes
ASM_TOKEN_PATTERN = re.compile(r'''
    0X[0-9A-F]+H(?![A-Z0-9_])  |   # Hexadecimal immediate, ensure H is not followed by letter or number
    [0-9A-F]+H(?![A-Z0-9_])    |   # Immediate like 80H, ensure H is not followed by letter or number
    \d+                        |   # Decimal number
    [A-Z_][A-Z0-9_]*           |   # Mnemonic / register / flag
    [\+\-\*/:\[\]\{\},]            # Operators and punctuation
''', re.VERBOSE)

def tokenize_asm_string(asm: str):
    """
    Tokenize assembly instruction string, supporting the following symbols:
    - Space
    - Comma
    - Square brackets []
    - Operators +-*/:
    - Immediate normalization
    
    Parameters:
        asm: assembly instruction string, e.g. "MOV\tEAX, [EBX+4*ECX+10h]"
        
    Returns:
        tokens: list of tokens after tokenization
    """
    asm = asm.upper().strip()
    # Try to split instruction by tab, if no tab then whole string is opcode
    parts = asm.split("\t", 1)
    opcode = parts[0]
    
    # If there is operand part, split by comma; otherwise operands is empty list
    oprands = parts[1].split(",") if len(parts) > 1 else []

    # Use precompiled regex for tokenization
    opcode_tokens = ASM_TOKEN_PATTERN.findall(opcode)
    oprands_tokens = [ASM_TOKEN_PATTERN.findall(oprand) for oprand in oprands]

    # Immediate normalization
    def normalize_token(tok):
        if tok in STRING_TO_REGISTER.keys():
            return tok
        try:
            if tok.startswith("0X"):
                val = int(tok.replace("H", ""), 16)
            elif tok.endswith("H"):
                val = int(tok[:-1], 16)
            elif tok.isdigit():
                val = int(tok)
            else:
                return tok
                
            # Values less than 128 are uniformly converted to decimal string representation
            # Values greater than or equal to 128 are normalized to IMM
            return "IMM"
        except:
            return tok
    
    opcode_tokens = [normalize_token(tok) for tok in opcode_tokens if tok.strip()]
    
    processed_oprands_tokens = []
    for oprand_tokens in oprands_tokens:
        # processed_tokens = [normalize_token(tok) for tok in oprand_tokens if tok.strip() and tok not in [",", "[", "]", ":", "{", "}"]]
        processed_tokens = [normalize_token(tok) for tok in oprand_tokens if tok.strip() and tok not in [",", "[", "]"]]
        processed_oprands_tokens.append(processed_tokens)

    return opcode_tokens, processed_oprands_tokens


def get_eflag_bit_name(bit):
    """
    Map RflagsBits.* values in iced-x86 to descriptive strings.
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
        return "[PAD]"

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
        return "[PAD]"

    return ",".join(updated_flags)


def tokenize_binary_instruction(binary_data: bytes, ip: int):
    """
    Convert binary instruction encoding and IP address to token list
    
    Parameters:
        binary_data: Binary instruction data
        ip: Instruction IP address
        
    Returns:
        tokens: Tokenized token list
    """
    # Initialize formatter
    info_factory = InstructionInfoFactory()
    formatter = Formatter(FormatterSyntax.NASM)
    formatter.gas_show_mnemonic_size_suffix = True
    
    # Decode binary data using iced_x86
    decoder = Decoder(64, binary_data, ip=ip, options=DecoderOptions.NONE)
    
    all_tokens = []
    offset = 0
    
    # Decode all instructions
    while offset < len(binary_data):
        old_position = decoder.position  # Record the position before decoding
        instruction = decoder.decode()
        info = info_factory.info(instruction)
        op_code = instruction.op_code()
        instruction_bytes = binary_data[old_position:decoder.position]
        
        # Check if the decoder has advanced, if not then break the loop
        if decoder.position == old_position:
            break
        
        # Get the disassembly representation of the instruction
        mnemonic_str = formatter.format_mnemonic(instruction, FormatMnemonicOptions.NO_PREFIXES)
        operands_str = formatter.format_all_operands(instruction)
        disasm = f"{mnemonic_str}\t{operands_str}"
        filter_list = [Code.NOP_RM32, Code.NOPW, Code.NOP_RM16]
        if instruction.code in filter_list or mnemonic_str.upper() == "NOP":
            disasm = "NOP\t"

        # Use the existing tokenization function to process the disassembled instruction
        opcode_tokens, oprands_tokens = tokenize_asm_string(disasm)
        if len(opcode_tokens) != 1:
            print(f"Warning: Number of tokens after disassembly is not 1 - Instruction: {disasm}, Tokens: {opcode_tokens}")
        assert len(opcode_tokens) == 1

        # Flatten the operand token list and merge with opcode tokens
        flat_oprands = []
        for oprand in oprands_tokens:
            flat_oprands.extend(oprand)
        asm_layer_embs = opcode_tokens + flat_oprands
        if instruction.code not in filter_list and len(oprands_tokens) != instruction.op_count:
            # Define valid operand count matching conditions
            valid_operand_conditions = [
                instruction.op_count == len(oprands_tokens),  # Normal case: parsed count equals actual count
                instruction.op_count == 3 and len(oprands_tokens) == 2,  # Special case 1: instruction has 3 operands but parsed 2
                instruction.op_count == 2 and len(oprands_tokens) == 1,
                len(oprands_tokens) == 0,   # Special case 2: no operands parsed
                len(oprands_tokens) > instruction.op_count,
                instruction.op_count == 4 and len(oprands_tokens) == 3 and code_to_string(instruction.code).endswith(("IMM8", "IMM8_SAE"))
            ]
            
            # If not satisfied any valid conditions, then report error
            if not any(valid_operand_conditions):
                # print the bytes of instruction
                print(instruction_bytes)
                print(f"Warning: Operand count mismatch - Instruction: {disasm}, parsed: {len(oprands_tokens)}, actual: {instruction.op_count}, op_code: {op_code.op_count}")
                for op in range(instruction.op_count):
                    print(f"Operand {op}: {op_code_operand_kind_to_string(op_code.op_kind(op))}")
                print(f"Operand tokens: {oprands_tokens}")
                print(code_to_string(instruction.code))
                print(f"Used registers: {[register_to_string(used_reg_info.register) for used_reg_info in info.used_registers()]}")
                exit(0)

        type_layer_embs = [("[PAD]", 1)]
        access_layer_embs = [("[PAD]", 1)]
        reg_layer_embs = ["[PAD]"]
        register_in_asm = []
        for i in range(len(oprands_tokens)):
            if len(oprands_tokens) == 2 and instruction.op_count == 3 and i == 1:
                index_op = i + 1
            else:
                index_op = i
            
            oprand_type = op_code_operand_kind_to_string(op_code.op_kind(index_op)).upper().strip()
            if oprand_type in ["NONE", "NULL"]:
                oprand_type = "[PAD]"
            type_layer_embs.append((oprand_type, len(oprands_tokens[i])))
            access_type = op_access_to_string(info.op_access(index_op)).upper().strip()
            if access_type in ["NONE", "NULL"]:
                access_type = "[PAD]"
            access_layer_embs.append((access_type, len(oprands_tokens[i])))

            for tok_index in range(len(oprands_tokens[i])):
                if oprands_tokens[i][tok_index] in STRING_TO_REGISTER.keys():
                    reg = STRING_TO_REGISTER[oprands_tokens[i][tok_index]]
                    full_reg_from_token = RegisterInfo(reg).full_register
                    updated = False
                    for used_reg_info in info.used_registers():
                        full_reg = RegisterInfo(used_reg_info.register).full_register
                        if full_reg == full_reg_from_token:
                            reg_layer_embs.append(register_to_string(used_reg_info.register))
                            register_in_asm.append(full_reg)
                            updated = True
                            break
                    if not updated:
                        reg_layer_embs.append(register_to_string(reg))
                        print(f"A case: {disasm} that has no matching register: {register_to_string(reg)}, used registers: {[register_to_string(used_reg_info.register) for used_reg_info in info.used_registers()]}")
                else:
                    reg_layer_embs.append("[PAD]")

        assert len(asm_layer_embs) == len(reg_layer_embs)
        eflag_layer_embs = get_eflags_string(instruction)
        if eflag_layer_embs in ["NONE", "NULL"]:
            eflag_layer_embs = "[PAD]"
        mnemonic_layer_embs = mnemonic_to_string(instruction.mnemonic)
        implicit_regs = []
        register_in_asm = set(register_in_asm)
        for reg_info in info.used_registers():
            if RegisterInfo(reg_info.register).full_register not in register_in_asm:
                implicit_regs.append(reg_info)
        implicit_regs_filter = {"STOSD", "MOVSB", "CDQE", "LEAVE", "MOVSD", "STOSQ", "DIV", "MOVSQ", "MOVSW", "CDQ", "IMUL", "IDIV", "CQO", "STOSB", "STOSW", "CBW", "MUL", "CMPXCHG", "CWDE", "VZEROUPPER", "FXCH", "FADDP", "FSUBRP", "FMULP", "FDIVRP", "FSUBP", "FDIVP", "CPUID", "XGETBV", "RDTSC", "RDTSCP", "SYSCALL", "FPREM", "CMPXCHG16B", "FCOM", "FCOMP", "CMPSB", "XSAVE", "XSTORE", "XSAVEC", "SCASB", "FCOMPP", "FUCOMPP", "LODSB", "FPREM1", "FSCALE", "FPATAN", "VZEROALL", "LODSD", "ENTER", "FYL2X", "XCRYPTECB", "XSHA512", "XSHA1", "XCRYPTCBC", "XSHA256", "XRSTOR", "SCASD", "FYL2XP1", "XLATB", "INSB", "OUTSB", "INSD", "OUTSD", "IRET", "CMPSQ", "CMPSD", "FNSAVE"}
        if len(implicit_regs) == 1:
            reg_layer_embs[0] = register_to_string(implicit_regs[0].register)
            access_layer_embs[0] = (op_access_to_string(implicit_regs[0].access), access_layer_embs[0][1])
        elif len(implicit_regs) > 1 and mnemonic_str.upper() not in implicit_regs_filter:
            # Print instruction information for debugging
            print(f"Warning: Implicit register count greater than 1 - Instruction: {disasm}, {instruction_bytes}")
            print(f"All registers: {[register_to_string(reg.register) for reg in info.used_registers()]}")
            print(f"Implicit registers: {[register_to_string(reg.register) for reg in implicit_regs]}")
            print(f"Used registers: {[register_to_string(used_reg) for used_reg in register_in_asm]}")
        
        token_results = {
            "asm": asm_layer_embs,
            "mne": mnemonic_layer_embs,
            "type": type_layer_embs,
            "reg": reg_layer_embs,
            "rw": access_layer_embs,
            "eflag": eflag_layer_embs
        }
        
        all_tokens.append(token_results)
        
        # Move to next instruction
        offset = decoder.position
    
    return all_tokens


if __name__ == "__main__":
    # Simple test
    test_cases = [
        "MOV\tEAX, 42",
        "ADD\tEBX, 0x7F",
        "SUB\tECX, 80h",
        "MOV\t[EBX+4*ECX+10h], EDX",
        "JMP\t0x1000",
        "CMP\tBYTE PTR [ESI], 0",
        "PUSH\t127",
        "POP\tEAX",
    ]
    
    print("Testing tokenization function:")
    for test in test_cases:
        tokens = tokenize_asm_string(test)
        print(f"Instruction: {test}")
        print(f"Tokenization result: {tokens}")
        print("-" * 50)
    
    
    # Example bytecode
    example_bytes = (
        b"\x48\x89\x5C\x24\x10\x48\x89\x74\x24\x18\x55\x57\x41\x56\x48\x8D"
        b"\xAC\x24\x00\xFF\xFF\xFF\x48\x81\xEC\x00\x02\x00\x00\x48\x8B\x05"
        b"\x18\x57\x0A\x00\x48\x33\xC4\x48\x89\x85\xF0\x00\x00\x00\x4C\x8B"
        b"\x05\x2F\x24\x0A\x00\x48\x8D\x05\x78\x7C\x04\x00\x33\xFF\x90\xf3\xab"
        b"Hi\xc9VUUU\xf3\xa4H\x8d\x04@E\x84\xf6\x87\xd6H\xf7\xe1\xf0\x0f\xb1\n\xdf\xf1\x86\xf6b\xf3uH\x1f\xcc\x04fA\x0f8\x10\xcb"
    )
    
    # Set a base IP address for decoding
    base_ip = 0x1000
    
    # Call function for testing
    tokens = tokenize_binary_instruction(example_bytes, base_ip)
    
    # Print results
    print("Decoding results:")
    for i, token_result in enumerate(tokens):
        print(f"\nInstruction {i+1}:")
        print(f"  Assembly layer: {token_result['asm']}")
        print(f"  Type layer: {token_result['type']}")
        print(f"  Register layer: {token_result['reg']}")
        print(f"  Read/write layer: {token_result['rw']}")
        print(f"  Flag layer: {token_result['eflag']}")