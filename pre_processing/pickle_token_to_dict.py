import pickle
import argparse
import os
import glob

control_tokens = [
    "[PAD]",
    "[SEP]",
    "[CLS]",
    "[UNK]",
    "[MASK]",
]


def main(input_path):
    # Get all pickle files in the directory
    pickle_files = glob.glob(os.path.join(input_path, "*.pkl"))

    for pickle_file in pickle_files:
        # Read pickle file content
        with open(pickle_file, "rb") as f:
            token_list = pickle.load(f)

        # Ensure token_list is an ordered list
        if not isinstance(token_list, list):
            token_list = sorted(list(token_list))

        # Concatenate control_tokens at the head of the list
        token_list = control_tokens + token_list

        # Create txt file with same name
        txt_file = os.path.splitext(pickle_file)[0] + ".txt"

        # Write list content line by line to txt file
        with open(txt_file, "w", encoding="utf-8") as f:
            for token in token_list:
                f.write(str(token) + "\n")

    # Create reg_r_tokens.txt and reg_w_tokens.txt
    reg_tokens = ["0", "1"]
    token_list = control_tokens + reg_tokens
    # Write to reg_r_tokens.txt
    with open(os.path.join(input_path, "reg_r_tokens.txt"), "w", encoding="utf-8") as f:
        for token in token_list:
            f.write(str(token) + "\n")

    # Write to reg_w_tokens.txt
    with open(os.path.join(input_path, "reg_w_tokens.txt"), "w", encoding="utf-8") as f:
        for token in token_list:
            f.write(str(token) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the directory containing pickle files",
    )
    args = parser.parse_args()

    main(args.input_path)

    print("Done!")
