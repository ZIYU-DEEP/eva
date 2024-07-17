"""
Adaptively sample and evolve the prompts.
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
                        default="cat-searcher/responses-gemma-1.1-2b-it-split-0-all-hf-rewards",
                        help='The dataset with prompts to evolve.')
    parser.add_argument("--output_dataset", type=str, 
                        default="cat-searcher/responses-gemma-1.1-2b-it-split-0-all-hf-rewards-resample-evol",
                        help='The evolved dataset.')
    
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--gen_model_name", type=str, default="gpt-4-turbo")
    parser.add_argument("--num_evolutions", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=20)
    
    parser.add_argument("--do_adaptive_sample", type=int, default=1,
                        choices=[0, 1], 
                        help="Adaptively sample for an informative subsets. 0 if not.")
    parser.add_argument("--sample_metric", type=str, default='reward_mean')
    parser.add_argument("--sample_frac", type=float, default=0.25)
    parser.add_argument("--sample_method", type=str, default='importance_weighted')

    return parser.parse_args()


def evolve_chunk(instructions, gen_model_name, num_evolutions):
    # Get the llm
    llm = OpenAILLM(model=gen_model_name)
    llm.generation_kwargs = {"max_new_tokens": 512}  # TODO: note this mean lead to api error
    
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


def adaptive_sample(
    input_dataset: str = 'cat-searcher/responses-gemma-1.1-2b-it-split-0-all-hf-rewards',
    hf_username: str = 'cat-searcher',
    sample_metric: str = 'reward_mean',
    sample_frac: float = 0.25,
    sample_method: str = 'importance_weighted'
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
        values = df[sample_metric]
        min_value, max_value = values.min(), values.max()
        normalized_values = (values - min_value) / (max_value - min_value)
        inverted_weights = 1 - normalized_values
        weights = inverted_weights / inverted_weights.sum()
        
    elif sample_metric in ['reward_var', 'reward_gap']:
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
    ds_name = f"{hf_username}/{input_dataset}-re-sample-{sample_metric}-{sample_method}-{sample_frac}"
    new_dataset.push_to_hub(
        ds_name, 
        split="train", 
        private=True
    )
    
    return ds_name


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
    
    evolve_dir = Path(data_root) / 'evolved'
    evolve_dir.mkdir(parents=True, exist_ok=True)
    csv_path = str(evolve_dir / f'{input_dataset.split("/")[-1]}.csv')
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Get the dataset
    dataset = load_dataset(input_dataset, split='train')
    # dataset = dataset.select(range(0, 50))  # DEBUG: this line is for debug
    
    # Create an informative subset
    if do_adaptive_sample:
        input_dataset = adaptive_sample(
            input_dataset=input_dataset,
            hf_username=hf_username,
            sample_metric=sample_metric,
            sample_frac=sample_frac,
            sample_method=sample_method
        )
        dataset = load_dataset(input_dataset, split='train')
        
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
                                   num_evolutions=num_evolutions)
    
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
