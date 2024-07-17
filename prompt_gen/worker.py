import pandas as pd
import argparse
import os

from multiprocessing import Pool
from datasets import load_dataset, Dataset
from distilabel.llms import OpenAILLM
from distilabel.steps.tasks import EvolInstruct
from pathlib import Path


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str, 
                        default="cat-searcher/responses-gemma-1.1-2b-it-split-0-all-hf-rewards-resample",
                        help='The dataset with prompts to evolve.')
    parser.add_argument("--output_dataset", type=str, 
                        default="cat-searcher/responses-gemma-1.1-2b-it-split-0-all-hf-rewards-resample-evol",
                        help='The evolved dataset.')
    parser.add_argument("--data_root", type=str, 
                        default="./data")
    parser.add_argument("--gen_model_name", type=str, default="gpt-4-turbo")
    parser.add_argument("--num_evolutions", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=40)


    return parser.parse_args()

# --------------------------------------------------------
# Parse the argument
args = parse_arguments()

data_root = args.data_root
input_dataset = args.input_dataset
output_dataset = args.output_dataset
num_workers = args.num_workers
gen_model_name = args.gen_model_name 
num_evolutions = args.num_evolutions

evolve_dir = Path(data_root) / 'evolved'
evolve_dir.mkdir(parents=True, exist_ok=True)
csv_path = str(evolve_dir / f'{input_dataset.split('/')[-1]}.csv')


# --------------------------------------------------------
# Get the dataset
dataset = load_dataset(args.input_dataset, split='train')
dataset_chunk = dataset.select(range(0, len(dataset)))  # In debug mode, use the any chunk
instruction_list = [{'instruction': prompt} for prompt in dataset_chunk['prompt']]
# --------------------------------------------------------


# Function to process a chunk of instructions
def process_chunk(instructions,
                  gen_model_name: str = 'gpt-4-turbo',
                  num_evolutions: int = 4):
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
# Process the chunks in parallel
with Pool(num_workers) as pool:
    result_chunks = pool.map(process_chunk, instruction_chunks)

# Flatten the list of results
prompt_list = [prompt for sublist in result_chunks for prompt in sublist]
# --------------------------------------------------------

# --------------------------------------------------------
# Make it a dataframe
df_chunk = pd.DataFrame()
df_chunk['prompt'] = prompt_list

# Save and push it
df_chunk.to_csv(, index=False)
repo = Dataset.from_csv('temp.csv')
repo.push_to_hub(output_dataset, split='train')
# --------------------------------------------------------
