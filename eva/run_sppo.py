#!/usr/bin/env python
#
# Adapted from https://github.com/huggingface/alignment-handbook

import logging
import random
import sys
import yaml

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    SPPOConfig,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)

from peft import PeftConfig, PeftModel
from trainer import SPPOTrainer

logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

def setup_logging(log_level):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

def load_and_process_datasets(data_args, tokenizer):
    raw_datasets = get_datasets(data_args, splits=["train"])
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)
    column_names = [x for x in column_names if x not in ['chosen_probs', 'chosen_probs_win', 'chosen_probs_lose']]

    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "skip_system_message": True},
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    for split in ["train"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", 
             "text_chosen": "chosen", 
             "text_rejected": "rejected"}
        )

    return raw_datasets

def setup_model(model_args, training_args):
    torch_dtype = (
        model_args.torch_dtype 
        if model_args.torch_dtype in ["auto", None] 
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision):
        logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            use_flash_attention_2=model_args.use_flash_attention_2,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            revision=model_args.model_revision,
        )
        model_kwargs = None

    ref_model = model
    ref_model_kwargs = model_kwargs

    if model_args.use_peft:
        ref_model = None
        ref_model_kwargs = None

    return model, ref_model, model_kwargs, ref_model_kwargs

def train_and_evaluate(trainer, raw_datasets, training_args):

    # -------------------------------------------------------------------------------- 
    # Set up the checkpoint
    last_checkpoint = get_checkpoint(training_args)
    
    # Set up the checkpoint
    # Train from a specified checkpoint given in the recipe
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        logger.info(f"Resuming training from the specified checkpoint.")
        
    # Train from the auto-saved checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")
    
    else:
        checkpoint = None
        logger.info(f"No checkpoint with {last_checkpoint}. Starting from scratch.")
    
    # Set the train result
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    # --------------------------------------------------------------------------------
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

def save_model_and_results(trainer, training_args, model_args, data_args):
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    trainer.accelerator.wait_for_everyone()
    logger.info("*** Training complete! ***")

def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SPPOConfig))
    model_args, data_args, training_args = parser.parse()
    training_args.do_eval = False
    num_iteration = 1

    try:
        for i in range(num_iteration):
            main_inner(model_args, data_args, training_args)
            print(f"-------------------------Finished Iteration {i+1}---------------------------------")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

def main_inner(model_args, data_args, training_args):
    setup_logging(training_args.get_process_log_level())

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # last_checkpoint = get_checkpoint(training_args)
    # if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #     logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")
    # else:
    #     logger.info(f"No checkpoint with {last_checkpoint}. Starting from scratch.")

    set_seed(training_args.seed)

    data_args.truncation_side = "left"
    tokenizer = get_tokenizer(model_args, data_args)
    raw_datasets = load_and_process_datasets(data_args, tokenizer)

    model, ref_model, model_kwargs, ref_model_kwargs = setup_model(model_args, training_args)

    trainer = SPPOTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],  # Do Not Eval For Now
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
        loss_type=training_args.loss_type,
    )

    train_and_evaluate(trainer, raw_datasets, training_args)
    save_model_and_results(trainer, training_args, model_args, data_args)


if __name__ == "__main__":
    main()
