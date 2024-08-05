# README

<!-- ```
src/
├── preload.py           # Preloads necessary data and models

├── generate.py          # Generates response pairs
├── combine_generate.py  # Combines generation results
├── compute_prob.py      # Computes probabilities for ranking
├── rank.py              # Ranks generated responses

├── pipeline.sh          # Main pipeline script for training

└── update_config.py    # Updates the dataset with new entries
``` -->

## Folder Structure
### Generating responses from a prompt set
```
├── generate.py          # Generates response pairs
├── combine_generate.py  # Combines generation results
├── rank.py              # Ranks generated responses
├── compute_prob.py      # Computes probabilities for ranking
```
Given a prompt set and a model:
- `generate.py`: generate response pairs and save in the json formats
- `combine_generate.py`: 
    - this is slightly silly as we could do things with DDP.
    - anyway, we now put the the j-th response for all prompts in response_j.json.
- `rank.py`:
    - taken prompts (list[str]) and responses (list[tuple]), generate a numpy array for ranks
    - where ransks[i][j] represents the rank of the j-th response for the i-th prompt.
- `compute_prob.py`:
    - this will push the generated responses to the hub
    - one with the suffix `all`, containing all the responses
    - one with the suffix `pair`, containing only the best and the worst
