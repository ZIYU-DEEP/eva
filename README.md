
# README

## About
We present a new asymmetric self-play framework for self-improving language model alignment.


## Environment Setup
1. **Setup:**
   ```bash
   # Set the conda environment
   conda create -n align python=3.10
   conda activate align

   # Set your own working directory
   root="~/github"  # Set your own working directory
   mkdir -p ${root}
   ```

2. **Download and Install Training Dependencies:**
   ```bash
   # Install general requirements
   cd ${root}
   git clone https://github.com/ziyu-deep/eva.git
   cd eva
   pip install -e ".[all]"

   # Install flashinfer 
   # Check https://docs.flashinfer.ai/installation.html for your version
   pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

   # Install PairRM (one option among other RM choices)
   cd ${root}
   git clone https://github.com/yuchenlin/LLM-Blender.git
   cd LLM-Blender
   pip install -e .
   ```

3. **Misc:**
   ```bash
   huggingface-cli login       
   wandb login  
   export WANDB_PROJECT=EVA               
   export OPENAI_API_KEY="..." 
   export GEMINI_API_KEY="..."
   export HF_API_KEY="..."

   # To run VLLM with Gemma-2 models, we have tested with the following setup:
   # Version 1
   pip install vllm==0.5.3
   wget https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.1/flashinfer-0.1.1+cu121torch2.3-cp310-cp310-linux_x86_64.whl
   pip install flashinfer-0.1.1+cu121torch2.3-cp310-cp310-linux_x86_64.whl
   export VLLM_ATTENTION_BACKEND=FLASHINFER

   # Version 2
   pip install vllm==0.5.4
   pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
   ```

## Training Scripts
Execute the training scripts based on the base model you choose. Be sure to modify the iteration number during the training.

1. Default training
   ```bash
   cd ${root}/eva
   bash ./scripts/gemma-2-9b-it-dpo/iterate.sh
   ```

2. Evolving training (example for one iteration)
   <!-- ```bash
   bash ./scripts/gemma-9b/evolve-create-iter-1.sh
   bash ./scripts/gemma-9b/evolve-gen-iter-1.sh
   bash ./scripts/gemma-9b/evolve-train-iter-1.sh
   ``` -->
   ```bash
   cd ${root}/eva
   bash ./scripts/gemma-2-9b-it-dpo/evolve-1.sh
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
  # Install relevant packages
  cd ${root}
  git clone https://github.com/lm-sys/FastChat.git
  cd FastChat
  pip install -e ".[model_worker,llm_judge]"
  pip install openai==1.40.3  # Notice we modify the original code

  # Run evaluation
  cd ${root}/eva/benchmark/mt_bench
  bash ./run.sh  # Be sure to modify the models to compare with
  ```

- Arena Hard
   ```bash
   cd ${root}/eva/benchmark/arena_hard
   bash ./scripts/serve.sh # Better run in separate screens
   bash ./scripts/run.sh  
   ```

- Alpaca Eval 2.0
   ```bash
   # Install relevant packages
   pip install alpaca-eval==0.6.3

   # Run evaluation
   cd ${root}/eva/benchmark/alpaca_eval
   bash ./run.sh # Be sure to modify the models to compare with
   ```

- LLM Evaluation Harness
   ```bash
   # Install relevant packages
   cd ${root}
   git clone https://github.com/EleutherAI/lm-evaluation-harness
   cd lm-evaluation-harness  # tested with the 0.4.3 version
   pip install -e .

   # Run evaluation
   cd ${root}/eva/benchmark/lm-evaluation-harness
   bash run.sh
   ```

## TODO
The generate part is quite slow. Consider rewriting it.

<!-- ## TODO
- [ ] Fix the naming issue (more like a tree adding suffix)
- [x] Make a spreadsheet
- [ ] Create our own evaluation datasets
- [ ] Adding reward model candidates and make a correlation plots on different reward models
- [ ] Increasing number of generations (check the log linear relationship)
- [ ] Adding hints/critique to generate better responses to contrast
- [x] Fix the Arena-Hard Evaluation
- [x] Using absolute reward model for dpo -->

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

