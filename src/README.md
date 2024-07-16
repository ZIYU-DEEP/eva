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