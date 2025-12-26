import re
import argparse
import csv
import os
from collections import Counter
import concurrent.futures
from tqdm import tqdm

def extract_instructions(file_path):
    """
    Read complete assembly instructions from file
    
    Args:
        file_path: path to file containing assembly instructions
        
    Returns:
        list of all complete instructions and corresponding opcodes list
    """
    instructions = []
    opcodes = []
    
    # Add progress bar for file reading progress
    print(f"Reading file: {file_path}")
    line_count = 0
    with open(file_path, 'r') as f:
        # First count file lines to initialize progress bar
        for _ in f:
            line_count += 1
    
    with open(file_path, 'r') as f:
        for line in tqdm(f, total=line_count, desc="Reading instructions"):
            line = line.strip()
            if not line:
                continue
                
            # Save complete instruction
            instructions.append(line)
            
            # Extract opcode (first part of instruction, usually before space or tab)
            match = re.match(r'^([a-zA-Z0-9_]+)', line)
            if match:
                opcode = match.group(1)
                opcodes.append(opcode)
            
    return instructions, opcodes


# Define instruction categories
INSTRUCTION_CATEGORIES = {
    # 1. Data transfer instructions - register to register
    1: {
        'name': 'Data Transfer - Register to Register',
        'patterns': [
            r'^mov\s+\w+,\s*\w+$', r'^xchg\s+\w+,\s*\w+$', r'movzx\s+\w+,\s*\w+$', 
            r'movsx\s+\w+,\s*\w+$', r'movsxd\s+\w+,\s*\w+$', r'^cmov\w+\s+\w+,\s*\w+$'
        ],
        'specific_opcodes': [
            'bswap', 'movlhps', 'movhlps', 'vmovlhps', 'vmovhlps', 'vmovd', 'vmovq',
            'movd', 'movq'
        ]
    },
    
    # 2. Data transfer instructions - memory access
    2: {
        'name': 'Data Transfer - Memory Access',
        'patterns': [
            r'^mov\s+\w+,\s*\[', r'^mov\s+\[', r'movzx\s+\w+,\s*\[', r'movsx\s+\w+,\s*\[',
            r'movsxd\s+\w+,\s*\[', r'^cmov\w+\s+\w+,\s*\[', r'lea\s+\w+,\s*\['
        ],
        'specific_opcodes': [
            'movaps', 'movapd', 'movups', 'movupd', 'movdqa', 'movdqu', 'movhps', 'movhpd', 
            'movlps', 'movlpd', 'movsd', 'movss', 'movsldup', 'movshdup', 'movddup',
            'vmovaps', 'vmovapd', 'vmovups', 'vmovupd', 'vmovdqa', 'vmovdqu', 'vmovhps', 'vmovhpd',
            'vmovlps', 'vmovlpd', 'vmovsd', 'vmovss', 'vmovsldup', 'vmovshdup', 'vmovddup',
            'lddqu', 'vlddqu'
        ]
    },
    
    # 3. Data transfer instructions - immediate
    3: {
        'name': 'Data Transfer - Immediate',
        'patterns': [
            r'^mov\s+\w+,\s*[0-9]', r'^mov\s+\w+,\s*0x', r'^mov\s+\w+,\s*-',
            r'^mov\s+\w+,\s*\+', r'^mov\s+\w+,\s*\(', r'^mov\s+\w+,\s*offset'
        ],
        'specific_opcodes': []
    },
    
    # 4. Data transfer instructions - stack operations
    4: {
        'name': 'Data Transfer - Stack',
        'patterns': [
            r'^push', r'^pop'
        ],
        'specific_opcodes': [
            'push', 'pop', 'pusha', 'popa', 'pushf', 'popf', 'pushad', 'popad'
        ]
    },
    
    # 5. Arithmetic instructions - register operations
    5: {
        'name': 'Arithmetic - Register',
        'patterns': [
            r'^add\s+\w+,\s*\w+$', r'^sub\s+\w+,\s*\w+$', r'^mul\s+\w+$', r'^div\s+\w+$',
            r'^idiv\s+\w+$', r'^imul\s+\w+,\s*\w+$', r'^imul\s+\w+$', r'^inc\s+\w+$',
            r'^dec\s+\w+$', r'^neg\s+\w+$', r'^adc\s+\w+,\s*\w+$', r'^sbb\s+\w+,\s*\w+$'
        ],
        'specific_opcodes': [
            'vaddsd', 'vaddss', 'vaddps', 'vaddpd', 'vsubsd', 'vsubss', 'vsubps', 'vsubpd',
            'vmulsd', 'vmulss', 'vmulps', 'vmulpd', 'vdivsd', 'vdivss', 'vdivps', 'vdivpd',
            'paddb', 'paddw', 'paddd', 'paddq', 'psubb', 'psubw', 'psubd', 'psubq',
            'vpaddb', 'vpaddw', 'vpaddd', 'vpaddq', 'vpsubb', 'vpsubw', 'vpsubd', 'vpsubq',
            'pmuldq', 'pmulld', 'pmuludq', 'vpmuldq', 'vpmulld', 'vpmuludq',
            'paddsb', 'paddsw', 'psubsb', 'psubsw', 'paddusb', 'paddusw', 'psubusb', 'psubusw',
            'vpaddsb', 'vpaddsw', 'vpsubsb', 'vpsubsw', 'vpaddusb', 'vpaddusw', 'vpsubusb', 'vpsubusw'
        ]
    },
    
    # 6. Arithmetic instructions - memory operations
    6: {
        'name': 'Arithmetic - Memory',
        'patterns': [
            r'^add\s+\w+,\s*\[', r'^add\s+\[', r'^sub\s+\w+,\s*\[', r'^sub\s+\[',
            r'^mul\s+\[', r'^div\s+\[', r'^idiv\s+\[', r'^imul\s+\w+,\s*\[', r'^imul\s+\[',
            r'^inc\s+\[', r'^dec\s+\[', r'^neg\s+\[', r'^adc\s+\w+,\s*\[', r'^adc\s+\[',
            r'^sbb\s+\w+,\s*\[', r'^sbb\s+\['
        ],
        'specific_opcodes': [
            'addsd', 'addss', 'addps', 'addpd', 'subsd', 'subss', 'subps', 'subpd',
            'mulsd', 'mulss', 'mulps', 'mulpd', 'divsd', 'divss', 'divps', 'divpd'
        ]
    },
    
    # 7. Arithmetic instructions - immediate
    7: {
        'name': 'Arithmetic - Immediate',
        'patterns': [
            r'^add\s+\w+,\s*[0-9]', r'^add\s+\w+,\s*0x', r'^add\s+\w+,\s*-',
            r'^sub\s+\w+,\s*[0-9]', r'^sub\s+\w+,\s*0x', r'^sub\s+\w+,\s*-',
            r'^imul\s+\w+,\s*\w+,\s*[0-9]', r'^imul\s+\w+,\s*[0-9]',
            r'^adc\s+\w+,\s*[0-9]', r'^sbb\s+\w+,\s*[0-9]'
        ],
        'specific_opcodes': []
    },
    
    # 8. Logical instructions - register operations
    8: {
        'name': 'Logical - Register',
        'patterns': [
            r'^and\s+\w+,\s*\w+$', r'^or\s+\w+,\s*\w+$', r'^xor\s+\w+,\s*\w+$',
            r'^not\s+\w+$', r'^test\s+\w+,\s*\w+$', r'^pand\s+\w+,\s*\w+$',
            r'^por\s+\w+,\s*\w+$', r'^pxor\s+\w+,\s*\w+$', r'^andn\s+\w+,\s*\w+,\s*\w+$'
        ],
        'specific_opcodes': [
            'vandps', 'vandpd', 'vorps', 'vorpd', 'vxorps', 'vxorpd',
            'andnps', 'andnpd', 'vandnps', 'vandnpd',
            'pandn', 'vpandn', 'vpandq', 'vpxord', 'vpxorq', 'vpord', 'vporq'
        ]
    },
    
    # 9. Logical instructions - memory operations
    9: {
        'name': 'Logical - Memory',
        'patterns': [
            r'^and\s+\w+,\s*\[', r'^and\s+\[', r'^or\s+\w+,\s*\[', r'^or\s+\[',
            r'^xor\s+\w+,\s*\[', r'^xor\s+\[', r'^not\s+\[', r'^test\s+\w+,\s*\[',
            r'^test\s+\[', r'^pand\s+\w+,\s*\[', r'^por\s+\w+,\s*\[', r'^pxor\s+\w+,\s*\['
        ],
        'specific_opcodes': [
            'andps', 'andpd', 'orps', 'orpd', 'xorps', 'xorpd'
        ]
    },
    
    # 10. Logical instructions - immediate
    10: {
        'name': 'Logical - Immediate',
        'patterns': [
            r'^and\s+\w+,\s*[0-9]', r'^and\s+\w+,\s*0x', r'^and\s+\w+,\s*-',
            r'^or\s+\w+,\s*[0-9]', r'^or\s+\w+,\s*0x', r'^or\s+\w+,\s*-',
            r'^xor\s+\w+,\s*[0-9]', r'^xor\s+\w+,\s*0x', r'^xor\s+\w+,\s*-',
            r'^test\s+\w+,\s*[0-9]', r'^test\s+\w+,\s*0x', r'^test\s+\w+,\s*-'
        ],
        'specific_opcodes': []
    },
    
    # 11. Control flow instructions
    11: {
        'name': 'Control Flow',
        'patterns': [
            r'^j', r'^call', r'^ret', r'^loop', r'^syscall', r'^int', r'^enter', r'^leave'
        ],
        'specific_opcodes': [
            'jmp', 'jz', 'jnz', 'ja', 'jb', 'jbe', 'jg', 'jge', 'jl', 'jle', 'jnb', 'js', 'jns', 
            'jo', 'jno', 'jp', 'jnp', 'jrcxz', 'call', 'retn', 'ret', 'retf', 'loop', 'syscall', 
            'int', 'enter', 'leave', 'bnd'
        ]
    },
    
    # 12. Comparison instructions - register operations
    12: {
        'name': 'Comparison - Register',
        'patterns': [
            r'^cmp\s+\w+,\s*\w+$', r'^test\s+\w+,\s*\w+$', r'^ucomis\w+\s+\w+,\s*\w+$',
            r'^comis\w+\s+\w+,\s*\w+$', r'^vcomis\w+\s+\w+,\s*\w+$'
        ],
        'specific_opcodes': [
            'vcomisd', 'vcomiss',
            'pcmpeqb', 'pcmpeqw', 'pcmpeqd', 'pcmpgtb', 'pcmpgtw', 'pcmpgtd',
            'vpcmpeqb', 'vpcmpeqw', 'vpcmpeqd', 'vpcmpgtb', 'vpcmpgtw', 'vpcmpgtd',
            'cmpeqps', 'cmpltps', 'cmpgtps', 'vcmpltps', 'vcmpgtps', 'vcmpeqps'
        ]
    },
    
    # 13. Comparison instructions - memory operations
    13: {
        'name': 'Comparison - Memory',
        'patterns': [
            r'^cmp\s+\w+,\s*\[', r'^cmp\s+\[', r'^test\s+\w+,\s*\[', r'^test\s+\['
        ],
        'specific_opcodes': [
            'ucomisd', 'ucomiss', 'comisd', 'comiss'
        ]
    },
    
    # 14. Comparison instructions - immediate
    14: {
        'name': 'Comparison - Immediate',
        'patterns': [
            r'^cmp\s+\w+,\s*[0-9]', r'^cmp\s+\w+,\s*0x', r'^cmp\s+\w+,\s*-',
            r'^test\s+\w+,\s*[0-9]', r'^test\s+\w+,\s*0x', r'^test\s+\w+,\s*-'
        ],
        'specific_opcodes': []
    },
    
    # 15. Bit shift and rotate instructions - register operations
    15: {
        'name': 'Shift and Rotate - Register',
        'patterns': [
            r'^shl\s+\w+,\s*\w+', r'^shr\s+\w+,\s*\w+', r'^sal\s+\w+,\s*\w+', r'^sar\s+\w+,\s*\w+',
            r'^rol\s+\w+,\s*\w+', r'^ror\s+\w+,\s*\w+', r'^shld\s+\w+,\s*\w+,\s*\w+', r'^shrd\s+\w+,\s*\w+,\s*\w+',
            r'^psll\w+\s+\w+,\s*\w+', r'^psrl\w+\s+\w+,\s*\w+', r'^psra\w+\s+\w+,\s*\w+'
        ],
        'specific_opcodes': [
            'shlx', 'shrx', 'sarx',
            'pslld', 'psllw', 'psllq', 'psrld', 'psrlw', 'psrlq', 'psrad', 'psraw',
            'vpslld', 'vpsllw', 'vpsllq', 'vpsrld', 'vpsrlw', 'vpsrlq', 'vpsrad', 'vpsraw',
            'vpsllvd', 'vpsllvq', 'vpsrlvd', 'vpsrlvq', 'vpsravd', 'vprorq', 'vprolq'
        ]
    },
    
    # 16. Bit shift and rotate instructions - memory operations
    16: {
        'name': 'Shift and Rotate - Memory',
        'patterns': [
            r'^shl\s+\[', r'^shr\s+\[', r'^sal\s+\[', r'^sar\s+\[',
            r'^rol\s+\[', r'^ror\s+\[', r'^shld\s+\w+,\s*\w+,\s*\[', r'^shrd\s+\w+,\s*\w+,\s*\['
        ],
        'specific_opcodes': []
    },
    
    # 17. Bit shift and rotate instructions - immediate
    17: {
        'name': 'Shift and Rotate - Immediate',
        'patterns': [
            r'^shl\s+\w+,\s*[0-9]', r'^shr\s+\w+,\s*[0-9]', r'^sal\s+\w+,\s*[0-9]', r'^sar\s+\w+,\s*[0-9]',
            r'^rol\s+\w+,\s*[0-9]', r'^ror\s+\w+,\s*[0-9]', r'^shld\s+\w+,\s*\w+,\s*[0-9]', r'^shrd\s+\w+,\s*\w+,\s*[0-9]'
        ],
        'specific_opcodes': []
    },
    
    # 18. SIMD and vector instructions
    18: {
        'name': 'SIMD and Vector',
        'patterns': [
            r'^p', r'^vp', r'^unpck', r'^vpunpck', r'^pack', r'^vpack',
            r'^shufp', r'^vshufp', r'^pshuf', r'^vpshuf', r'^palignr', r'^vpalignr',
            r'^punpck', r'^perm', r'^vperm'
        ],
        'specific_opcodes': [
            'addps', 'subps', 'mulps', 'divps', 'maxps', 'minps',
            'vaddps', 'vsubps', 'vmulps', 'vdivps', 'vmaxps', 'vminps',
            'packuswb', 'packssdw', 'vpackuswb', 'vpackssdw',
            'punpckhbw', 'punpcklbw', 'punpckhdq', 'punpckldq', 'punpckhqdq', 'punpcklqdq',
            'vpunpckhbw', 'vpunpcklbw', 'vpunpckhdq', 'vpunpckldq', 'vpunpckhqdq', 'vpunpcklqdq',
            'shufps', 'shufpd', 'vshufps', 'vshufpd', 'pshufb', 'pshufd', 'vpshufb', 'vpshufd',
            'vperm2f128', 'vperm2i128', 'vpermd', 'vpermq', 'vpermt2d', 'vpermt2ps',
            'vbroadcastss', 'vbroadcastsd', 'vbroadcastf128', 'vpbroadcastd',
            'haddpd', 'haddps', 'vhaddpd', 'vhaddps',
            'insertps', 'vinsertps', 'extractps', 'vextractps',
            'pinsrb', 'pinsrw', 'pinsrd', 'pextrb', 'pextrw', 'pextrd',
            'vpinsrb', 'vpinsrw', 'vpinsrd', 'vpextrb', 'vpextrw', 'vpextrd',
            'pblendw', 'blendps', 'blendpd', 'vpblendw', 'vblendps', 'vblendpd',
            'blendvps', 'blendvpd', 'pblendvb', 'vblendvps', 'vblendvpd', 'vpblendvb'
        ]
    },
    
    # 20. System instructions
    20: {
        'name': 'System',
        'patterns': [
            r'^sys', r'^cpuid', r'^rdtsc', r'^xsave', r'^xrstor', r'fence', r'^hlt',
            r'^cli', r'^sti', r'^in', r'^out', r'^clflush'
        ],
        'specific_opcodes': [
            'syscall', 'cpuid', 'rdtsc', 'rdrand', 'rdseed', 'xsave', 'xrstor', 'xsavec', 
            'lfence', 'mfence', 'sfence', 'clflush', 'clc', 'cld', 'cli', 'sti', 'hlt',
            'in', 'out', 'wait', 'pause', 'ud2', 'icebp', 'xgetbv', 'stmxcsr', 'ldmxcsr',
            'fxsave', 'fxrstor', 'endbr64'
        ]
    },
    
    # 21. Conditional set instructions
    21: {
        'name': 'Conditional Set',
        'patterns': [
            r'^set'
        ],
        'specific_opcodes': [
            'setz', 'setnz', 'seta', 'setb', 'setbe', 'setg', 'setl', 'setle', 'setnb',
            'setnbe', 'setnl', 'setnle', 'sets', 'setns', 'seto', 'setno', 'setp', 'setnp'
        ]
    },
    
    # 22. String operation instructions
    22: {
        'name': 'String Operations',
        'patterns': [
            r'^movs', r'^stos', r'^scas', r'^rep', r'^repe'
        ],
        'specific_opcodes': [
            'movsb', 'movsw', 'movsd', 'movsq', 'stosb', 'stosw', 'stosd', 'stosq',
            'scasb', 'scasw', 'scasd', 'scasq', 'rep', 'repe'
        ]
    },
    
    # 23. Floating point instructions (x87)
    23: {
        'name': 'x87 Floating Point',
        'patterns': [
            r'^f'
        ],
        'specific_opcodes': [
            'fadd', 'fsub', 'fmul', 'fdiv', 'fld', 'fst', 'fstp', 'fcom', 'fcomp',
            'fild', 'fist', 'fistp', 'fldcw', 'fstcw', 'fnstcw', 'fldenv', 'fstenv', 'fnstenv',
            'fstsw', 'fnstsw', 'fcomi', 'fcomip', 'fucomi', 'fucomip', 'fabs', 'fchs',
            'fscale', 'fsqrt', 'fyl2x', 'f2xm1', 'fldz', 'fld1', 'fprem', 'fxch', 'fxam'
        ]
    },
    
    # 24. Advanced SIMD instructions (AVX-512, FMA, advanced bitwise operations)
    24: {
        'name': 'Advanced SIMD',
        'patterns': [
            r'^vfmadd', r'^vfmsub', r'^vfnmadd', r'^vfnmsub', r'^vbroadcast', r'^vextract',
            r'^vgather', r'^vscatter', r'^pdep', r'^pext', r'^lzcnt', r'^tzcnt', r'^popcnt',
            r'^bzhi', r'^blsr', r'^blsi', r'^blsmsk', r'^rorx'
        ],
        'specific_opcodes': [
            'vfmadd132pd', 'vfmadd132ps', 'vfmadd132sd', 'vfmadd132ss',
            'vfmadd213pd', 'vfmadd213ps', 'vfmadd213sd', 'vfmadd213ss',
            'vfmadd231pd', 'vfmadd231ps', 'vfmadd231sd', 'vfmadd231ss',
            'vfmsub132pd', 'vfmsub132ps', 'vfmsub132sd', 'vfmsub132ss',
            'vfnmadd132pd', 'vfnmadd132ps', 'vfnmadd132sd', 'vfnmadd132ss',
            'vfnmsub132pd', 'vfnmsub132ps', 'vfnmsub132sd', 'vfnmsub132ss',
            'vbroadcastf32x4', 'vbroadcastf32x8', 'vextractf32x4', 'vextracti32x8',
            'vgatherdps', 'vpgatherdd', 'vgatherqps', 'vpgatherqd',
            'pdep', 'pext', 'lzcnt', 'tzcnt', 'popcnt', 'bzhi', 'blsr', 'rorx',
            'vpmulhrsw', 'vpmuludq', 'vgatherdps', 'vpternlogd'
        ]
    },
    
    # 25. Conversion instructions
    25: {
        'name': 'Conversion',
        'patterns': [
            r'^cvt', r'^vcvt', r'^cbw', r'^cwde', r'^cdq', r'^cdqe', r'^cqo'
        ],
        'specific_opcodes': [
            'cvtsi2sd', 'cvtsd2ss', 'cvttsd2si', 'cvtsi2ss', 'cvtss2sd', 'cvttss2si',
            'vcvtsi2sd', 'vcvtsd2ss', 'vcvttsd2si', 'vcvtsi2ss', 'vcvtss2sd', 'vcvttss2si',
            'cvtps2pd', 'cvtpd2ps', 'cvtdq2ps', 'cvtps2dq', 'cvttps2dq', 'cvtpd2dq', 'cvtdq2pd',
            'vcvtps2pd', 'vcvtpd2ps', 'vcvtdq2ps', 'vcvtps2dq', 'vcvttps2dq', 'vcvtpd2dq', 'vcvtdq2pd',
            'cbw', 'cwde', 'cdq', 'cdqe', 'cqo'
        ]
    },
    
    # 26. Atomic operations
    26: {
        'name': 'Atomic Operations',
        'patterns': [
            r'^lock', r'^xadd', r'^cmpxchg'
        ],
        'specific_opcodes': [
            'lock', 'xadd', 'cmpxchg', 'xchg'
        ]
    },
    
    # 27. Miscellaneous instructions (difficult to categorize)
    27: {
        'name': 'Miscellaneous',
        'patterns': [
            r'^k', r'^vzeroupper'
        ],
        'specific_opcodes': [
            'kmovb', 'kmovw', 'kmovd', 'kmovq', 'knotd', 'knotq',
            'kandb', 'kandw', 'kandnb', 'kandnw', 'korw', 'kord',
            'kxnorb', 'kxnorw', 'kxorb', 'kunpckbw', 'kunpckwd', 'kunpckdq',
            'kortestb', 'kortestd', 'vzeroupper'
        ]
    }
}


def precompile_patterns():
    """Pre-compile all regex patterns to improve performance"""
    for category_id, category_info in INSTRUCTION_CATEGORIES.items():
        compiled_patterns = []
        for pattern in category_info['patterns']:
            compiled_patterns.append(re.compile(pattern))
        INSTRUCTION_CATEGORIES[category_id]['compiled_patterns'] = compiled_patterns

def classify_instruction(instruction):
    """Assign category ID for given complete instruction"""
    # Extract opcode part (without operands)
    opcode = instruction.split()[0] if ' ' in instruction else instruction
    
    for category_id, category_info in INSTRUCTION_CATEGORIES.items():
        # First check specific opcode list
        if opcode in category_info['specific_opcodes']:
            return category_id
            
        # Then check pattern matching (including operands)
        for pattern in category_info['compiled_patterns']:
            if pattern.match(instruction):
                return category_id
                
    # If no match found, return a default category
    return 0  # 0 means uncategorized


def process_chunk(chunk):
    """Process instruction block, return category list"""
    return [classify_instruction(instruction) for instruction in chunk]


def process_instructions(instructions, num_workers=None):
    """
    Process list containing complete instructions and assign category ID to each instruction
    Use multi-threading for acceleration
    
    Args:
        instructions: instruction list
        num_workers: number of worker threads, default None (use CPU core count)
        
    Returns:
        category list and category statistics
    """
    # Pre-compile regex patterns
    precompile_patterns()
    
    # Determine number of worker threads
    if num_workers is None:
        import multiprocessing
        num_workers = multiprocessing.cpu_count()
    
    # Calculate number of instructions processed by each thread
    total_instructions = len(instructions)
    chunk_size = max(1, total_instructions // (num_workers * 4))  # Each thread processes multiple small chunks to balance load
    
    categories = []
    
    # Use thread pool for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Split instructions into chunks
        chunks = [instructions[i:i+chunk_size] for i in range(0, total_instructions, chunk_size)]
        
        # Submit tasks and show progress bar
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        
        # Collect results and show progress
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Classifying instructions"):
            categories.extend(future.result())
    
    # Count number of instructions in each category and percentage
    category_stats = {}
    total_count = len(categories)
    
    # Count number of instructions in each category
    category_counter = Counter(categories)
    
    # Initialize category statistics
    for cat_id, count in category_counter.items():
        category_name = INSTRUCTION_CATEGORIES.get(cat_id, {'name': 'Unknown'})['name']
        category_stats[cat_id] = {
            'name': category_name,
            'count': count,
            'percentage': (count / total_count) * 100
        }
    
    return categories, category_stats

def main():
    parser = argparse.ArgumentParser(description='Count instruction types in assembly instruction file')
    parser.add_argument('input_file', help='path to file containing assembly instructions', default='unique_instructions.txt', nargs='?')
    parser.add_argument('-t', '--top', type=int, help='show top N opcodes with highest frequency', default=20)
    parser.add_argument('-o', '--output', help='write results to specified output file')
    parser.add_argument('-c', '--category', help='output instruction category statistics', action='store_true')
    parser.add_argument('-j', '--jobs', type=int, help='number of threads for parallel processing', default=None)
    
    args = parser.parse_args()
    
    # Extract complete instructions and opcodes
    instructions, opcodes = extract_instructions(args.input_file)
    
    # Classify instructions
    print(f"Starting classification of {len(instructions)} instructions...")
    categories, category_stats = process_instructions(instructions, num_workers=args.jobs)
    
    # Write category IDs to file
    print("Saving classification results...")
    with open("./inst_cate.txt", 'w') as file:
        for cat_id in tqdm(categories, desc="Writing category IDs"):
            file.write(f"{cat_id}\n")
    
    # Count number of different opcodes
    opcode_counter = Counter(opcodes)
    total_opcodes = len(opcode_counter)
    total_instructions = len(opcodes)
    
    # Prepare output results
    output_lines = [
        f"Total {total_opcodes} different opcodes",
        f"Total {total_instructions} assembly instructions",
        "\nMost common opcodes:",
    ]
    
    # Add most common opcodes and their occurrence counts
    for opcode, count in opcode_counter.most_common(args.top):
        percentage = (count / total_instructions) * 100
        output_lines.append(f"{opcode}: {count} ({percentage:.2f}%)")
    
    # If category statistics need to be output
    if args.category:
        output_lines.append("\nInstruction category statistics:")
        for cat_id, stats in sorted(category_stats.items()):
            output_lines.append(f"{cat_id}. {stats['name']}: {stats['count']} ({stats['percentage']:.2f}%)")
    
    # Output results
    output_text = "\n".join(output_lines)
    print(output_text)
    
    # If output file is specified, write results to file
    if args.output:
        print(f"Writing detailed statistics to: {args.output}")
        with open(args.output, 'w') as f:
            f.write(output_text)
            f.write("\n\nAll opcodes and their occurrence counts:\n")
            for opcode, count in sorted(opcode_counter.items()):
                percentage = (count / total_instructions) * 100
                f.write(f"{opcode}: {count} ({percentage:.2f}%)\n")
            
            if args.category:
                f.write("\n\nDetailed instruction category statistics:\n")
                for cat_id, stats in sorted(category_stats.items()):
                    f.write(f"{cat_id}. {stats['name']}: {stats['count']} ({stats['percentage']:.2f}%)\n")
                    
        print(f"Detailed statistics written to: {args.output}")


if __name__ == "__main__":
    main()