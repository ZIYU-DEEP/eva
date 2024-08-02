
# README

## About
We present a new self-play framework for language model alignment and a new learning objective derived from the self-play framework to align large language models more efficiently and effectively.


## Environment Setup
1. **Create a Virtual Environment:**
   ```bash
   conda create -n align python=3.10
   conda activate align
   ```

2. **Install PairRM:** (a temporary choice for RM during training)
   ```bash
   git clone https://github.com/yuchenlin/LLM-Blender.git
   cd LLM-Blender
   pip install -e .
   cd ..
   ```

3. **Download and Install Training Dependencies:**
   ```bash
   git clone https://github.com/ziyu-deep/eva.git
   cd eva
   pip install -e .
   ```

4. **Misc:**
   ```bash
   huggingface-cli login       
   wandb login                 
   export OPENAI_API_KEY="..."  # get a key at platform.openai.com/api-keys

   # To run VLLM with Gemma-2 models, we currently use the following setup:
   cd ..

   pip install vllm==0.5.3
   wget https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.1/flashinfer-0.1.1+cu121torch2.3-cp310-cp310-linux_x86_64.whl
   pip install flashinfer-0.1.1+cu121torch2.3-cp310-cp310-linux_x86_64.whl
   export VLLM_ATTENTION_BACKEND=FLASHINFER

   cd eva
   ```

## Training Scripts (TEMP)
Execute the training scripts based on the base model you choose. Be sure to modify the iteration number during the training.

1. Default training
   ```bash
   bash ./scripts/default/generate.sh
   bash ./scripts/default/train.sh 
   ```

2. Evolving training
   ```bash
   # Evaluate the prompt quality
   bash ./scripts/default/generate.sh
   python ./src/reward_hf.py

   # Generate new prompt set
   python ./data_gen/evol_prompt.py
   python ./data_gen/combine_df.py

   # Standard training
   bash ./scripts/evolve/train.sh 
   ```

To run the gemma baseline, do:
```
bash ./scripts/gemma-baseline/iterate.sh
```

<!-- - Generation for Y|X:
  ```bash
  bash ./scripts/generate.sh
  ```

- (Optional) Evolving for X'|X:
  ```bash
  bash ./scripts/evolve_x.sh
  ```

- Training:
  ```bash
  # use the raw X
  bash ./scripts/train.sh 

  # use X + X'
  bash ./scripts/train_plus.sh
  ``` -->

## Evaluation on Benchmarks
See detailed instructions for different benchmarks in `./benchmark`.

- MT Bench:
  ```bash
  cd ./benchmark/mt_bench
  bash ./run.sh  # Be sure to modify the models to compare with
  ```

- Arena Hard
   ```bash
   cd ./benchmark/arena_hard
   bash ./run.sh # Be sure to modify the models to compare with
   ```

## TODO
- [ ] Fix the naming issue (more like a tree adding suffix)

<!-- - Alpaca Eval
   ```bash
   cd ./benchmark/arena_hard
   bash ./run.sh # Be sure to modify the models to compare with
   ``` -->

<!-- ### Breakdown of Scripts:
1. **Generation:**
   ```bash
   python scripts/generate.py --model $MODEL --maxlen 2048 --output_dir $OUTPUT_DIR --prompts $PROMPTS
   ```
Main parameters:
- `model`: Specifies the model used for generation. In the first iteration, the model should be either `mistralai/Mistral-7B-Instruct-v0.2` or `meta-llama/Meta-Llama-3-8B-Instruct`.
- `maxlen`: Sets the token length for generation, defining the maximum number of tokens generated.
- `pairs`: Determines the number of generated samples per prompt, with a default setting of 5. Please note that changing this number is not supported by the overall pipeline.
- `output_dir`: Specifies the directory paths for saving intermediate results.
- `prompts`: Defines the set of prompts used for generation.
- `frac_len`: Enables the operation of vllm on multiple GPUs by dividing prompts into different fractions. `frac_len` defines the number of prompts in each fraction. For usage examples, see `generate.sh`.
- `data_frac`: Used in conjunction with `frac_len` for multi-GPU setups, `data_frac` indicates which fraction of the data the current GPU is processing. Refer to `generate.sh` for more details.


2. **Ranking:**
   ```bash
   python scripts/rank.py --output_dir $OUTPUT_DIR --prompts $PROMPTS
   ```
Main Parameters:
- `output_dir`: Specifies the directory paths where intermediate results are saved. Note that the default script attempts to push datasets to Hugging Face under the UCLA-AGI organization. You may need to adjust this to your organization, obtain write access for UCLA-AGI, or disable the `push_to_hub` command if necessary.
- `pairs`: Sets the number of generated samples per prompt, with a default of 5. Please note that other numbers are not supported by the overall pipeline.
- `frac_len`: This parameter is used to enable the use of PairRM on multiple GPUs by dividing prompts into different fractions. `frac_len` determines the number of prompts in each fraction. For usage examples, refer to `generate.sh`.
- `data_frac`: Similar to `frac_len`, this option is used for running PairRM on multiple GPUs. It specifies which fraction of the data the current GPU is processing. See `generate.sh` for examples.
- `prompts`: Defines the set of prompts used for generation.
- `gpu`: Indicates the GPU index used for ranking; it should match the `data_frac` parameter.

3. **Training:**
   ```bash
   bash scripts/pipeline.sh --model $MODEL --iter $ITER --dataset $DATASET --output_dir $OUTPUT_DIR --num 1
   ```
Main Parameters:
- model: The base model for training.
- dataset: The dataset used for training.
- output_dir: The name of the output model.
- num: The number of training epochs. -->

<!-- ## Evaluation
We adhere to the established guidelines for evaluation and utilize the following repositories:
- [AlpacaEval 2](https://github.com/tatsu-lab/alpaca_eval)
- [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)
- [HuggingFace Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)

We provide the model configurations used during AlpacaEval 2 in the `models_configs` directory. Please note that after the initial release of our model, we retrained it using a slightly modified prompt. The win rates observed post-retraining are comparable to the original results. -->


## Acknowledgements
This repo has referenced [SPPO](https://github.com/uclaml/sppo), [The Alignment Handbook](https://github.com/huggingface/alignment-handbook), [TRL](https://github.com/huggingface/trl), [FastChat](https://github.com/lm-sys/FastChat), [PairRM](https://github.com/yuchenlin/LLM-Blender), [vllm](https://github.com/vllm-project/vllm), [distilabel](https://distilabel.argilla.io/1.2.1/).

