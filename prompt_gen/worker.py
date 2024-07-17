import pandas as pd
import sys
from datasets import load_dataset, Dataset
from distilabel.llms import OpenAILLM
from distilabel.steps.tasks import EvolInstruct

def evolve_prompt(prompt):
    llm = OpenAILLM(model='gpt-4-turbo')
    evol_instruct = EvolInstruct(
        llm=llm,
        num_evolutions=4,
        store_evolutions=True,
    )
    evol_instruct.load()
    result = next(evol_instruct.process([{"instruction": prompt}]))
    evolved_instructions = result['evolved_instructions']
    return [{'prompt': prompt, 'evolved_instruction': instr} for instr in evolved_instructions]

def process_chunk(start_idx, end_idx, temp_csv_path):
    # Load the dataset
    dataset = load_dataset('cat-searcher/test', split='train')

    # Select the chunk
    dataset_chunk = dataset.select(range(start_idx, end_idx))

    # Process the chunk
    results = []
    for prompt in dataset_chunk['prompt']:
        results.extend(evolve_prompt(prompt))

    # Convert the results to a DataFrame
    df = pd.DataFrame(results)

    # Save the DataFrame to a temporary CSV file
    df.to_csv(temp_csv_path, index=False)

if __name__ == "__main__":
    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    temp_csv_path = sys.argv[3]

    process_chunk(start_idx, end_idx, temp_csv_path)
