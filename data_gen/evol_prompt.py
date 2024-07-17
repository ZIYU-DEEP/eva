"""
Evolve the prompts.
"""

from multiprocessing import Pool
from datasets import load_dataset, Dataset
from distilabel.llms import OpenAILLM
from distilabel.steps.tasks import EvolInstruct
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import argparse
import os


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str, 
                        default="cat-searcher/responses-gemma-1.1-2b-it-split-0-all-hf-rewards",
                        help='The dataset with prompts to evolve.')
    parser.add_argument("--output_dataset", type=str, 
                        default="cat-searcher/responses-gemma-1.1-2b-it-split-0-all-hf-rewards-resample-evol",
                        help='The evolved dataset.')
    parser.add_argument("--hf_username", type=str, default="cat-searcher")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--gen_model_name", type=str, default="gpt-4-turbo")
    parser.add_argument("--num_evolutions", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=40)

    return parser.parse_args()



def evolve_chunk(instructions, gen_model_name, num_evolutions):
    # Get the llm
    llm = OpenAILLM(model=gen_model_name)
    
    # Create the task for evolving instructions
    evol_instruct = EvolInstruct(
        llm=llm,
        num_evolutions=num_evolutions,
        store_evolutions=True,
    )
    
    # Load the task
    evol_instruct.load()
    
    # Generate the results
    result_list = next(evol_instruct.process(instructions))
    
    # Extract evolved instructions
    evolved_prompts = []
    for result in result_list:
        for result_i in result['evolved_instructions']:
            evolved_prompts.append(result_i)
    
    return evolved_prompts


def prompt_sample(
    input_dataset: str,
    hf_username: str = 'cat-searcher',
    metric: str = 'reward_mean',
    frac: float = 0.25,
    sampling_method: str = 'importance_weighted'
) -> None:
    """
    Sample prompts based on a specified metric with conditional logic and advanced sampling.
    """
    # Load dataset
    if '/' in input_dataset: input_dataset = input_dataset.split('/')[-1]
    dataset = load_dataset(f"{hf_username}/{input_dataset}", split="train")
    df = dataset.to_pandas()

    # ----------------------------------------------------------------------------------
    # Calculate weights based on the metric
    if metric in ['reward_mean']:
        values = df[metric]
        min_value, max_value = values.min(), values.max()
        normalized_values = (values - min_value) / (max_value - min_value)
        inverted_weights = 1 - normalized_values
        weights = inverted_weights / inverted_weights.sum()
        
    elif metric in ['reward_var', 'reward_gap']:
        values = df[metric]
        min_value, max_value = values.min(), values.max()
        normalized_values = (values - min_value) / (max_value - min_value)
        weights = normalized_values / normalized_values.sum()
        
    else:
        raise ValueError(f"Metric {metric} not recognized or supported for sampling.")
    # ----------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------
    # Perform sampling based on the specified method
    if sampling_method == 'importance_weighted':
        # Simple importance weighted sampling
        sampled_df = df.sample(n=int(len(df) * frac), weights=weights, replace=False)
        
    elif sampling_method == 'stratified':
        # Stratified sampling based on the metric
        df['weights'] = weights
        sampled_df = df.groupby(pd.qcut(df[metric], q=10, duplicates='drop')).apply(
            lambda x: x.sample(frac=frac, weights=x['weights'], replace=False)).reset_index(drop=True)
    else:
        raise ValueError(f"Sampling method {sampling_method} not recognized.")
    # ----------------------------------------------------------------------------------

    # Save and push to hub
    new_dataset = Dataset.from_pandas(sampled_df)
    new_dataset.push_to_hub(
        f"{hf_username}/{input_dataset}-re-sample-{metric}-{sampling_method}-{frac}", 
        split="train", 
        private=True
    )


def main():
    # --------------------------------------------------------
    # Parse the arguments
    args = parse_arguments()

    data_root = args.data_root
    input_dataset = args.input_dataset
    output_dataset = args.output_dataset
    num_workers = args.num_workers
    gen_model_name = args.gen_model_name 
    num_evolutions = args.num_evolutions

    evolve_dir = Path(data_root) / 'evolved'
    evolve_dir.mkdir(parents=True, exist_ok=True)
    csv_path = str(evolve_dir / f'{input_dataset.split("/")[-1]}.csv')
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Get the dataset
    dataset = load_dataset(input_dataset, split='train')
    # dataset = dataset.select(range(0, 50))  # TODO: This line is for debug
    instruction_list = [{'instruction': prompt} for prompt in dataset['prompt']]
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Split instruction list into chunks
    chunk_size = len(instruction_list) // num_workers
    instruction_chunks = [instruction_list[i:i + chunk_size] 
                          for i in range(0, len(instruction_list), chunk_size)]

    if len(instruction_chunks[-1]) < chunk_size:  # Ensure the last chunk is not empty
        instruction_chunks[-2].extend(instruction_chunks[-1])
        instruction_chunks.pop()
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Process the chunks in parallel with progress bar
    with Pool(num_workers) as pool:
        result_chunks = list(tqdm(pool.starmap(
            evolve_chunk, 
            [(chunk, gen_model_name, num_evolutions) for chunk in instruction_chunks]), total=len(instruction_chunks)))

    # Flatten the list of results
    prompt_list = [prompt for sublist in result_chunks for prompt in sublist]
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Make it a dataframe
    df_chunk = pd.DataFrame()
    df_chunk['prompt'] = prompt_list

    # Save and push it
    df_chunk.to_csv(csv_path, index=False)
    repo = Dataset.from_csv(csv_path)
    repo.push_to_hub(output_dataset, split='train', private=True)
    print(f'Dataset pushed to huggingface.co/datasets/{output_dataset}.')
    # --------------------------------------------------------


if __name__ == "__main__":
    main()
