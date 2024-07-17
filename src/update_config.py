"""
Update the dataset_mixer, model_name_or_path, and hub_model_id entries in the config_full.yaml.
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
    parser.add_argument('--model_name', type=str,
                        default='google/gemma-1.1-2b-it')
    parser.add_argument('--hub_model_id', type=str,
                        default='cat-searcher/gemma-1.1-2b-it-sppo-iter0')
    
    return parser.parse_args()


def main():
    # Parse the arguments
    args = parse_arguments()
    
    # Read the original config
    with open(args.config_path, 'r') as file:
        content = file.read()

    # Replace the dataset_mixer content
    content = re.sub(
        re.compile(r'dataset_mixer:\n\s*[^:]+:\s*\d+(\.\d+)?'),  
        f'dataset_mixer:\n  {args.dataset}: 1.0', 
        content)
    
    # Replace the model_name_or_path
    content = re.sub(
        re.compile(r'model_name_or_path:\s*[^\s]+'),  
        f'model_name_or_path: {args.model_name}', 
        content)
    
    # Replace the hub_model_id
    content = re.sub(
        re.compile(r'hub_model_id:\s*[^\s]+'),  
        f'hub_model_id: {args.hub_model_id}', 
        content)
    
    # Replace the output_dir
    content = re.sub(
        re.compile(r'output_dir:\s*[^\s]+'),  
        f'output_dir: checkpoints/{args.hub_model_id.split("/")[1]}', 
        content)

    # Write the updated config
    with open(args.config_path, 'w') as file:
        file.write(content)

    print('Config updated successfully.')


if __name__ == "__main__":
    main()
