# Scripts to Train and Evaluate Chat Models

Reference: [Alignment Handbook](https://github.com/huggingface/alignment-handbook/blob/main/scripts/README.md)

## Fine-tuning

Here, `{task}` refers to the type of training you wish to run. Currently the following tasks are supported:
* continued pretraining `cpt` 
* supervised finetuning `sft`
* direct preference optimisation `dpo`
* odds ratio preference optimisation `orpo`

`{model_name}` refers to the choice of a recipe in the `recipes` directory. For example, to replicate Zephyr-7B-β you can run:

```shell
# Step 1 - train SFT policy
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/zephyr-7b-beta/sft/config_full.yaml

# Step 2 - align with DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/zephyr-7b-beta/dpo/config_full.yaml
```

**💡 Tip:** If you scale up/down the number of GPUs, we recommend also scaling up the per-device batch size or number of gradient accumulation steps to keep the global batch size constant (and thus replicate our results).

By default, these scripts will push each model to your Hugging Face Hub username, i.e. `{username}/{model_name}-{task}`. You can override the parameters in each YAML config by appending them to the command as follows:

```shell
# Change batch size, number of epochs etc
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_{task}.py recipes/{model_name}/{task}/config_full.yaml --per_device_train_batch_size=42 --num_train_epochs=5
```

## Logging with Weights and Biases

By default all training metrics are logged with TensorBoard. If you have a [Weights and Biases](https://wandb.ai/site) account and are logged in, you can view the training metrics by appending `--report_to=wandb`, e.g.

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_{task}.py recipes/{model_name}/{task}/config_full.yaml --report_to=wandb
```

## Fine-tuning on your datasets

Under the hood, each training script uses the `get_datasets()` function which allows one to easily combine multiple datasets with varying proportions. For instance, this is how one can specify multiple datasets and which splits to combine in one of the YAML configs:

```yaml
datasets_mixer:
    dataset_1: 0.5  # Use 50% of the training examples
    dataset_2: 0.66 # Use 66% of the training examples
    dataset_3: 0.10 # Use 10% of the training examples
dataset_splits:
- train_xxx         # The training splits to mix
- test_xxx          # The test splits to mix
```

If you want to fine-tune on your datasets, the main thing to keep in mind is how the chat templates are applied to the dataset blend. Since each task (SFT, DPO, ORPO, etc), requires a different format, we assume the datasets have the following columns:

**SFT**

* `messages`: A list of `dicts` in the form `{"role": "{role}", "content": {content}}`. 
* See [ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) for an example.

**DPO and ORPO**

* `chosen`: A list of `dicts` in the form `{"role": "{role}", "content": {content}}` corresponding to the preferred dialogue.
* `rejected`: A list of `dicts` in the form `{"role": "{role}", "content": {content}}` corresponding to the dispreferred dialogue.
* See [ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) for an example.

We also find it useful to include dedicated splits per task in our datasets, so e.g. we have:

* `{train,test}_sft`: Splits for SFT training.
* `{train,test}_gen`: Splits for generation ranking like rejection sampling or PPO.
* `{train,test}_prefs`: Splits for preference modelling, like reward modelling or DPO.

If you format your dataset in the same way, our training scripts should work out of the box!

## Evaluating chat models

We recommend benchmarking chat models on:

* [MT-Bench](https://huggingface.co/spaces/lmsys/mt-bench): a multi-turn benchmark spanning 80 dialogues and 10 domains.
* [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval): a single-turn benchmark which evaluates the helpfulness of chat and instruct models against `text-davinci-003`.

For both benchmarks, we have added support for the [Zephyr chat template](https://huggingface.co/alignment-handbook/zephyr-7b-sft-full/blob/ac6e600eefcce74f5e8bae1035d4f66019e93190/tokenizer_config.json#L30) (which is the default produced by our scripts), so you can evaluate models produced by our scripts as follows:

**MT-Bench**

* Follow the installation instructions [here](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)
* Make sure the word `zephyr` exists in the `--model-path` argument when generating the model responses [here](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge#step-1-generate-model-answers-to-mt-bench-questions). This will ensure the correct chat template is loaded. For example, the following model name is valid: `--model-path {hub_username}/my-baby-zephyr`
* Generate the model responses and GPT-4 rankings.

**AlpacaEval**

* Follow the installation instructions [here](https://github.com/tatsu-lab/alpaca_eval#quick-start)
* Copy-paste the [config](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/models_configs/zephyr-7b-beta/configs.yaml) for `zephyr-7b-beta` and place it in the `model_configs` directory under `{your_zephyr_model}`.
  * Next, update the [config name](https://github.com/tatsu-lab/alpaca_eval/blob/2daa6e11b194653043ca74f735728dc068e04aae/src/alpaca_eval/models_configs/zephyr-7b-beta/configs.yaml#L1) and [Hub model ID](https://github.com/tatsu-lab/alpaca_eval/blob/2daa6e11b194653043ca74f735728dc068e04aae/src/alpaca_eval/models_configs/zephyr-7b-beta/configs.yaml#L5) to match your model name.
* Follow the steps to evaluate your model [here](https://github.com/tatsu-lab/alpaca_eval/tree/main#evaluating-a-model).

Note that MT-Bench and AlpacaEval rely on LLMs like GPT-4 to judge the quality of the model responses, and thus the ranking exhibit various biases including a preference for models distilled from GPTs. For that reason, we also recommend submitting your best models for human evaluation in:

* [Chatbot Arena](https://chat.lmsys.org): a live, human evaluation of chat models in head-to-head comparisons.