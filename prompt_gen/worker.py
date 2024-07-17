from datasets import load_dataset, Dataset
from distilabel.llms import OpenAILLM
from distilabel.steps.tasks import EvolInstruct
import pandas as pd
from multiprocessing import Pool

# --------------------------------------------------------
# Get the dataset
dataset = load_dataset('cat-searcher/test', split='train')
dataset_chunk = dataset.select(range(0, 5))
instruction_list = [{'instruction': prompt} for prompt in dataset_chunk['prompt']]
# --------------------------------------------------------

# Function to process a chunk of instructions
def process_chunk(instructions):
    # Get the llm
    llm = OpenAILLM(model='gpt-4-turbo')
    
    # Create the task for evolving instructions
    evol_instruct = EvolInstruct(
        llm=llm,
        num_evolutions=1,
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
num_workers = 4  # You can adjust this number based on your system's capabilities
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
df_chunk.to_csv('temp.csv', index=False)
repo = Dataset.from_csv('temp.csv')
repo.push_to_hub('temp', split='train')
# --------------------------------------------------------
