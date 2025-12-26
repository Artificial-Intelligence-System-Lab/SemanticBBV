import pickle
import os
import argparse # Import argparse module

def inspect_pkl(file_path: str):
    """
    Load pickle file and print its data structure.

    Args:
        file_path: Path to the pickle file.
    """
    try:
        # Ensure file path exists
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print(f"Successfully loaded file: {file_path}")
        print("=" * 30)
        print("Data structure analysis:")
        print("=" * 30)

        data_type = type(data)
        print(f"Top-level data type: {data_type}")

        if isinstance(data, list):
            print(f"List length: {len(data)}")
            if len(data) > 0:
                first_element_type = type(data[0])
                print(f"Type of the first element in the list: {first_element_type}")
                if isinstance(data[0], dict):
                    print("Dictionary structure of the first element (keys and value types):")
                    for key, value in data[0].items():
                        print(f"  - Key '{key}' (type: {type(key)}): value type {type(value)}")
                        if isinstance(value, list) and len(value) > 0:
                            print(f"    - Type of the first element in the nested list: {type(value[0])}")
                        elif isinstance(value, dict) and len(value) > 0:
                             first_nested_key = next(iter(value))
                             print(f"    - First key in the nested dictionary '{first_nested_key}' (type: {type(first_nested_key)}): value type {type(value[first_nested_key])}")
                elif isinstance(data[0], list):
                    print(f"Length of the first element's list: {len(data[0])}")
                    if len(data[0]) > 0:
                        print(f"Type of the first sub-element in the first element's list: {type(data[0][0])}")
                    # New: Print the content of the first internal list
                    print("=" * 20)
                    print("Content of the first internal list:")
                    print(data[0])
                    print("=" * 20)
                elif isinstance(data[0], tuple):
                    print(f"Length of the first element's tuple: {len(data[0])}")
                    if len(data[0]) > 0:
                        print(f"Type of the first sub-element in the first element's tuple: {type(data[0][0])}")
                        if isinstance(data[0][0], list) and len(data[0][0]) > 0:
                            print(f"  - Length of the first list element in the tuple: {len(data[0][0])}")
                            print(f"  - Type of the first sub-element of the first list element in the tuple: {type(data[0][0][0])}")
                        elif isinstance(data[0][0], dict) and len(data[0][0]) > 0:
                            print("Dictionary structure of the first element in the tuple (keys and value types):")
                            for key, value in data[0][0].items():
                                print(f"  - Key '{key}' (type: {type(key)}): value type {type(value)}")
                    # Print the content of the first tuple
                    print("=" * 20)
                    print("Content of the first internal tuple:")
                    print(data[0])
                    print("=" * 20)
                

        elif isinstance(data, dict):
            print(f"Number of keys in the dictionary: {len(data)}")
            if len(data) > 0:
                first_key = next(iter(data)) # Get the first key
                first_value_type = type(data[first_key])
                print(f"Type of the value for the first key '{first_key}' (type: {type(first_key)}) in the dictionary: {first_value_type}")
                if isinstance(data[first_key], list) and len(data[first_key]) > 0:
                     print(f"  - Type of the first element in the value list for the first key: {type(data[first_key][0])}")
                elif isinstance(data[first_key], dict) and len(data[first_key]) > 0:
                     first_nested_key_in_dict = next(iter(data[first_key]))
                     print(f"  - First key in the value dictionary for the first key '{first_nested_key_in_dict}' (type: {type(first_nested_key_in_dict)}): value type {type(data[first_key][first_nested_key_in_dict])}")
        elif isinstance(data, tuple):
            print(f"Tuple length: {len(data)}")
            if len(data) > 0:
                print(f"Type of the first element in the tuple: {type(data[0])}")
                if isinstance(data[0], list) and len(data[0]) > 0:
                    print(f"  - Length of the first list element in the tuple: {len(data[0])}")
                    print(f"  - Type of the first sub-element of the first list element in the tuple: {type(data[0][0])}")
                    # Print the content of the first list
                    print("=" * 20)
                    print("Content of the first list in the tuple:")
                    print(data[0])
                    print("=" * 20)
                elif isinstance(data[0], dict) and len(data[0]) > 0:
                    print("Dictionary structure of the first element in the tuple (keys and value types):")
                    for key, value in data[0].items():
                        print(f"  - Key '{key}' (type: {type(key)}): value type {type(value)}")
                        if isinstance(value, list) and len(value) > 0:
                            print(f"    - Type of the first element in the nested list: {type(value[0])}")
                        elif isinstance(value, dict) and len(value) > 0:
                            first_nested_key = next(iter(value))
                            print(f"    - First key in the nested dictionary '{first_nested_key}' (type: {type(first_nested_key)}): value type {type(value[first_nested_key])}")
        
        # You can add handling for other data types as needed
        # For example: tuples, sets, etc.

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except pickle.UnpicklingError:
        print(f"Error: File '{file_path}' is not a valid pickle file or is corrupted.")
    except Exception as e:
        print(f"Unknown error occurred while loading or analyzing file '{file_path}': {e}")

if __name__ == "__main__":
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Load and analyze the data structure of a pickle file.")
    
    # Add a positional argument 'file_path' to receive the path of the pkl file
    parser.add_argument("file_path", help="Path to the pickle file to analyze")
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Call the inspect_pkl function using the file path obtained from the command line
    inspect_pkl(args.file_path)