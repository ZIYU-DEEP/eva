"""
Adaptively sample and evolve the prompts.
TODO: add stratified resampling.
"""

from multiprocessing import Pool
from datasets import load_dataset, Dataset
from distilabel.llms import OpenAILLM
from distilabel.steps.tasks import EvolInstruct
from pathlib import Path
from tqdm import tqdm
from functools import partial

import pandas as pd
import argparse
import time
import os


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--hf_username", type=str, default="cat-searcher")
    parser.add_argument("--input_dataset", type=str, 
                        default="cat-searcher/responses-gemma-1.1-2b-it-split-1-iter-1-all-hf-rewards",
                        help='The dataset with prompts to evolve.')
    parser.add_argument("--subset_dataset", type=str, 
                        default="cat-searcher/responses-gemma-1.1-2b-it-split-1-iter-1-resample-subset-reward_gap-0.25",
                        help='The evolved dataset.')
    parser.add_argument("--output_dataset", type=str, 
                        default="cat-searcher/responses-gemma-1.1-2b-it-split-1-iter-1-resample-evol-reward_gap-0.25",
                        help='The evolved dataset.')
    
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--gen_model_name", type=str, default="gpt-4-0125-preview")
    parser.add_argument("--num_evolutions", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--evolve_temperature", type=float, default=1.0)
    
    parser.add_argument("--do_adaptive_sample", type=int, default=1,
                        choices=[0, 1], 
                        help="Adaptively sample for an informative subsets. 0 if not.")
    parser.add_argument("--sample_metric", type=str, default='reward_mean')
    parser.add_argument("--sample_frac", type=float, default=0.25)
    parser.add_argument("--sample_method", type=str, default='importance_weighted')

    return parser.parse_args()


def evolve_chunk(instructions,
                 gen_model_name: str='gpt-4-0125-preview', 
                 num_evolutions: int=4, 
                 max_prompt_length: int=512,
                 evolve_temperature: float=1.0):
    # Get the llm
    llm = OpenAILLM(model=gen_model_name)
    llm.generation_kwargs = {
        "max_new_tokens": max_prompt_length,  # TODO: too large may lead to api error; original 2048
        "temperature": evolve_temperature,
        }  
    
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
    # for result in result_list:
    for result in tqdm(result_list, desc="Processing Instructions"):
        for result_i in result['evolved_instructions']:
            evolved_prompts.append(result_i)
    
    return evolved_prompts


def adaptive_sample(
    input_dataset: str = 'cat-searcher/responses-gemma-1.1-2b-it-split-0-all-hf-rewards',
    hf_username: str = 'cat-searcher',
    sample_metric: str = 'reward_mean',
    sample_frac: float = 0.25,
    sample_method: str = 'importance_weighted',
    subset_dataset: str = 'cat-searcher/responses-gemma-1.1-2b-it-split-0-subset-reward_gap-0.25',
) -> str:
    """
    Sample prompts based on a specified metric with conditional logic and advanced sampling.
    """
    # Load dataset
    if '/' in input_dataset: input_dataset = input_dataset.split('/')[-1]
    dataset = load_dataset(f"{hf_username}/{input_dataset}", split="train")
    df = dataset.to_pandas()

    # ----------------------------------------------------------------------------------
    # Calculate weights based on the metric
    if sample_metric in ['reward_mean']:
        # Encourage where you are weak
        values = df[sample_metric]
        min_value, max_value = values.min(), values.max()
        normalized_values = (values - min_value) / (max_value - min_value)
        inverted_weights = 1 - normalized_values
        weights = inverted_weights / inverted_weights.sum()
    
    elif sample_metric in ['reward_maxmean', 'reward_loo', 'reward_var', 'reward_gap']:
        # Encourage where we there are improvement room or more contrast
        values = df[sample_metric]
        min_value, max_value = values.min(), values.max()
        normalized_values = (values - min_value) / (max_value - min_value)
        weights = normalized_values / normalized_values.sum()
        
    else:
        raise ValueError(f"Metric {sample_metric} not recognized.")
    # ----------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------
    # Perform sampling based on the specified method
    if sample_method == 'importance_weighted':
        # Simple importance weighted sampling
        sampled_df = df.sample(n=int(len(df) * sample_frac), 
                               weights=weights, 
                               replace=False)
        
    elif sample_method == 'stratified':
        # Stratified sampling based on the metric
        df['weights'] = weights
        sampled_df = df.groupby(pd.qcut(df[sample_metric], 
                                        q=10,  # TODO: better use a parameter
                                        duplicates='drop')).apply(
            lambda x: x.sample(frac=sample_frac, 
                               weights=x['weights'], 
                               replace=False)).reset_index(drop=True)
    else:
        raise ValueError(f"Sampling method {sample_method} not recognized.")
    # ----------------------------------------------------------------------------------

    # Save and push to hub
    new_dataset = Dataset.from_pandas(sampled_df)
    new_dataset.push_to_hub(
        subset_dataset, 
        split="train", 
        private=True
    )
    
    return subset_dataset


def main():
    # --------------------------------------------------------
    # Parse the arguments
    start_time = time.time()
    args = parse_arguments()

    data_root = args.data_root
    input_dataset = args.input_dataset
    output_dataset = args.output_dataset
    num_workers = args.num_workers
    gen_model_name = args.gen_model_name 
    num_evolutions = args.num_evolutions
    
    do_adaptive_sample = args.do_adaptive_sample
    hf_username = args.hf_username
    sample_metric = args.sample_metric
    sample_frac = args.sample_frac
    sample_method = args.sample_method
    subset_dataset = args.subset_dataset
    max_prompt_length = args.max_prompt_length
    evolve_temperature = args.evolve_temperature
    
    evolve_dir = Path(data_root) / 'evolved'
    evolve_dir.mkdir(parents=True, exist_ok=True)
    csv_path = str(evolve_dir / f'{input_dataset.split("/")[-1]}.csv')
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Get the dataset
    dataset = load_dataset(input_dataset, split='train')
    
    # DEBUG
    # print('The original dataset.')
    # print(f"Dataset size: {len(dataset)}")
    # print(f"Dataset columns: {dataset.column_names}")
    # print(f"First few entries: {dataset[:1]}")
    # dataset = dataset.select(range(0, 50))  # DEBUG: this line is for debug
    
    # Create an informative subset
    if do_adaptive_sample:
        input_dataset = adaptive_sample(
            input_dataset=input_dataset,
            hf_username=hf_username,
            sample_metric=sample_metric,
            sample_frac=sample_frac,
            sample_method=sample_method,
            subset_dataset=subset_dataset,
        )
        dataset = load_dataset(input_dataset, split='train')
        
        print('The subsampling dataset.')
        print(f"Dataset size: {len(dataset)}")
        print(f"Dataset columns: {dataset.column_names}")
        print(f"First few entries: {dataset[:1]}")
    
    # dataset = dataset.select(range(0, 10))  # DEBUG: this line is for debug
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
    # Wrap evolve_chunk to handle multiple arguments
    evolve_chunk_wrapper = partial(evolve_chunk, 
                                   gen_model_name=gen_model_name, 
                                   num_evolutions=num_evolutions,
                                   max_prompt_length=max_prompt_length,
                                   evolve_temperature=evolve_temperature)
    
    # Process the chunks in parallel with progress bar
    with Pool(num_workers) as pool:
        result_chunks = []
        for result in tqdm(pool.imap_unordered(
                evolve_chunk_wrapper, 
                instruction_chunks), total=len(instruction_chunks)):
            result_chunks.append(result)
            
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
    
    elapsed_time = time.time() - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds.")
    # --------------------------------------------------------


if __name__ == "__main__":
    main()
