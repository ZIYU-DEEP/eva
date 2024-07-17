#!/bin/bash

# Load the dataset size
dataset_size=$(python -c "from datasets import load_dataset; dataset = load_dataset('cat-searcher/test', split='train'); print(len(dataset))")

# Number of chunks (adjust this as needed)
num_chunks=10
chunk_size=$((dataset_size / num_chunks))

# Directory for temporary CSV files
temp_dir="temp_csvs"
mkdir -p $temp_dir

# Run the worker script in parallel
for i in $(seq 0 $((num_chunks - 1))); do
    start_idx=$((i * chunk_size))
    end_idx=$((start_idx + chunk_size))

    if [ $i -eq $((num_chunks - 1)) ]; then
        end_idx=$dataset_size  # Ensure the last chunk includes any remaining data
    fi

    python worker.py $start_idx $end_idx "$temp_dir/chunk_$i.csv" &
done

# Wait for all background processes to finish
wait

# Combine all temporary CSV files into the final CSV file
final_csv="evolved_prompts.csv"
head -n 1 "$temp_dir/chunk_0.csv" > $final_csv  # Copy the header from the first file
for f in $temp_dir/chunk_*.csv; do
    tail -n +2 "$f" >> $final_csv  # Append the data, skipping the header
done

# Clean up temporary files
rm -r $temp_dir
