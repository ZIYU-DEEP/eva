
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
   ```bash
   cd ${root}/eva
   bash ./scripts/gemma-2-9b-it-dpo/evolve-1.sh
   ```

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

## Acknowledgements
This repo has referenced [SPPO](https://github.com/uclaml/sppo), [The Alignment Handbook](https://github.com/huggingface/alignment-handbook), [TRL](https://github.com/huggingface/trl), [FastChat](https://github.com/lm-sys/FastChat), [PairRM](https://github.com/yuchenlin/LLM-Blender), [vllm](https://github.com/vllm-project/vllm), [distilabel](https://distilabel.argilla.io/1.2.1/).

