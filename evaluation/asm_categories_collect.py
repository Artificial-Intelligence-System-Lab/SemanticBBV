import random
import os
import pickle
from collections import defaultdict

def sample_instructions_by_category(instructions_file, categories_file, encoding_path, output_dir="sampled_instructions", sample_size=1000, ignore_categories=None):
    """
    Randomly select specified number of assembly instructions from each category and output to new file
    
    Parameters:
    instructions_file -- file containing assembly instructions
    categories_file -- file containing instruction classification
    encoding_path -- file path containing instruction tokens
    output_dir -- output directory
    sample_size -- number of samples from each category
    ignore_categories -- list of categories to ignore, default None
    """
    # Initialize ignore categories list
    if ignore_categories is None:
        ignore_categories = ["0", "unknown"]
    
    # Read instructions and categories
    with open(instructions_file, 'r') as f_inst, open(categories_file, 'r') as f_cate:
        instructions = [line.strip() for line in f_inst]
        categories = [line.strip() for line in f_cate]
    
    encoding_dict = None
    if encoding_path and os.path.exists(encoding_path):
        with open(encoding_path, "rb") as f:
            encoding_dict = pickle.load(f)

    # Handle mismatch in line counts between two files
    if len(instructions) != len(categories):
        print(f"Warning: instructions file ({len(instructions)} lines) and categories file ({len(categories)} lines) have inconsistent line counts")
        print(f"Difference: {len(instructions) - len(categories)} lines")
        
        # Choose processing method
        if len(instructions) > len(categories):
            # If there are more instructions, generate default categories for excess instructions (using opcodes)
            print("Will use opcode as category for instructions lacking category")
            for i in range(len(categories), len(instructions)):
                instruction = instructions[i]
                # Extract opcode (usually first part of instruction)
                parts = instruction.split()
                opcode = parts[0] if parts else "unknown"
                categories.append(opcode)
        else:
            # If there are more categories, truncate excess categories
            print("Will truncate excess category information")
            categories = categories[:len(instructions)]
    
    # Organize instructions by category, ignoring specified categories
    instructions_by_category = defaultdict(list)
    for instruction, category in zip(instructions, categories):
        # Skip ignored categories
        if category in ignore_categories:
            continue
        
        if encoding_dict and instruction not in encoding_dict:
            continue
        instructions_by_category[category].append(instruction)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create overall output file
    combined_output_file = os.path.join(output_dir, "combined_sampled_instructions.txt")
    # Create corresponding category output file
    combined_categories_file = os.path.join(output_dir, "combined_sampled_categories.txt")
    
    
    # Sample and output for each category
    all_sampled_instructions = []
    all_sampled_categories = []
    
    print(f"Processing {len(instructions_by_category)} different categories of instructions...")
    print(f"Ignored categories: {ignore_categories}")
    
    for category, category_instructions in instructions_by_category.items():
        # Decide sample quantity
        actual_sample_size = min(sample_size, len(category_instructions))
        
        if actual_sample_size < sample_size:
            print(f"Warning: category {category} has only {len(category_instructions)} instructions, less than requested {sample_size}")
        
        # Sample instructions
        sampled = random.sample(category_instructions, actual_sample_size)
        
        # Add sampling results to total list
        all_sampled_instructions.extend(sampled)
        # Add corresponding category for each sampled instruction
        all_sampled_categories.extend([category] * actual_sample_size)
        
        # Output to separate category file
        category_output_file = os.path.join(output_dir, f"category_{category}_sampled.txt")
        with open(category_output_file, 'w') as f_out:
            for instruction in sampled:
                f_out.write(f"{instruction}\n")
        
        print(f"Category {category}: sampled {actual_sample_size} instructions to file {category_output_file}")
    
    # Output all sampled instructions to one file
    with open(combined_output_file, 'w') as f_out:
        for instruction in all_sampled_instructions:
            f_out.write(f"{instruction}\n")
    
    # Output categories corresponding to all sampled instructions to one file
    with open(combined_categories_file, 'w') as f_out:
        for category in all_sampled_categories:
            f_out.write(f"{category}\n")
    
    if encoding_dict:
        combined_tokens_file = os.path.join(output_dir, "combined_sampled_instructions.pkl")
        with open(combined_tokens_file, "wb") as f_out:
            embeddings = []
            for instruction in all_sampled_instructions:
                embeddings.append(encoding_dict[instruction])
            pickle.dump(embeddings, f_out)

    # Check if number of sampled instructions and categories are consistent
    if len(all_sampled_instructions) != len(all_sampled_categories):
        print(f"Error: number of sampled instructions ({len(all_sampled_instructions)}) does not match number of categories ({len(all_sampled_categories)})!")
        # Calculate sample quantity for each category for debugging
        category_counts = {}
        for category in all_sampled_categories:
            category_counts[category] = category_counts.get(category, 0) + 1
        print("Sample quantity for each category:")
        for category, count in category_counts.items():
            print(f"  {category}: {count}")
    
    print(f"Saved all sampled instructions (total {len(all_sampled_instructions)}) to file {combined_output_file}")
    print(f"Saved categories corresponding to all sampled instructions (total {len(all_sampled_categories)}) to file {combined_categories_file}")
    
    return instructions_by_category, all_sampled_instructions

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Randomly sample specified number of assembly instructions from each category")
    parser.add_argument("--instructions", "-i", default="./unique_instructions_iced.txt", 
                        help="Path to file containing assembly instructions")
    parser.add_argument("--categories", "-c", default="./inst_cate.txt",
                        help="Path to file containing instruction classification")
    parser.add_argument("--encoding_path", "-e", default="./unique_instructions_iced_encodings.pkl",
                        help="File path containing instruction tokens")
    parser.add_argument("--output", "-o", default="sampled_instructions", 
                        help="Output directory")
    parser.add_argument("--sample_size", "-n", type=int, default=200,
                        help="Number of samples from each category")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--ignore_categories", "-ic", nargs="+", default=["0", "unknown"],
                        help="List of categories to ignore, default ignores categories 0 and unknown")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Execute sampling
    categories_dict, sampled_instructions = sample_instructions_by_category(
        args.instructions, 
        args.categories, 
        args.encoding_path,
        args.output, 
        args.sample_size,
        args.ignore_categories
    )
    
    # Output statistics for each category
    print("\nCategory statistics:")
    for category, instructions in categories_dict.items():
        print(f"Category {category}: total {len(instructions)} instructions")
