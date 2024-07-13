"""
Combine response files from multiple GPUs into a single file.

Now we put the responses of each pair in responses_{j}.json,
where j is the response index.
"""

import json
import pandas as pd
import argparse
from pathlib import Path


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output_dir', type=str, 
                        default='responses-gemma-1.1-2b-it-split-0')
    parser.add_argument("--data_root", type=str, 
                        default="./data")
    parser.add_argument("--n_pairs", type=int, default=5)
    parser.add_argument("--n_gpus", type=int, default=8)
    return parser.parse_args()


def main():
    
    # Parse the arguments
    args = parse_arguments()

    output_dir = Path(args.output_dir)  # Shared across files
    data_root = Path(args.data_root)
    gen_dir = data_root / 'generated' / output_dir

    # Combine the responses
    for j in range(args.n_pairs):
        results = []
        for i in range(args.n_gpus):
            
            # Get the file for the j-th response from the i-th GPU
            file_path = gen_dir / f"responses_{i}_{j}.json"
            print(f'Reading from {file_path}')
            
            # Append the responses to the results
            with open(file_path) as f:
                gen = json.load(f)
                results += gen

        # Now the results from the j-th response are in a single file
        output_path = gen_dir / f"responses_{j}.json"

        print(f'Saved to {output_path}')
        with open(output_path, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
