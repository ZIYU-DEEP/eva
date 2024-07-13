"""
Update the dataset_mixer entry in the config_full.yaml.
"""
import re
import argparse


def parse_arguments():
    """
    Parse command line arguments.
    """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str,
                        default='responses-gemma-1.1-2b-it-split-0-pair')
    parser.add_argument('--config_path', type=str, 
                        default='recipes/default/config_full.yaml')
    
    return parser.parse_args()


def main():
    # Parse the arguments
    args = parse_arguments()
    
    # Read the original config
    with open(args.config_path, 'r') as file:
        content = file.read()

    # Replace the matched pattern with the new dataset_mixer content
    new_content = re.sub(
        re.compile(r'dataset_mixer:\n\s*[^:]+:\s*\d+(\.\d+)?'),  
        f'dataset_mixer:\n  {args.dataset}: 1.0', 
        content)

    # Write the updated config
    with open(args.config_path, 'w') as file:
        file.write(new_content)

    print('Dataset mixer updated successfully.')


if __name__ == "__main__":
    main()
