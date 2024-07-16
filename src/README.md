# README

```
src/
├── preload.py           # Preloads necessary data and models

├── generate.py          # Generates response pairs
├── combine_generate.py  # Combines generation results
├── compute_prob.py      # Computes probabilities for ranking
├── rank.py              # Ranks generated responses

├── pipeline.sh          # Main pipeline script for training

└── update_config.py    # Updates the dataset with new entries
```

## Folder Structure
### Generating responses from a prompt set
```
├── generate.py          # Generates response pairs
├── combine_generate.py  # Combines generation results
├── compute_prob.py      # Computes probabilities for ranking
├── rank.py              # Ranks generated responses
```
Given a prompt set and a model:
- `generate.py`: generate response paris and save in the json formats
- `combine_generate.py`: this is slightly silly as we could do things with DDP.
