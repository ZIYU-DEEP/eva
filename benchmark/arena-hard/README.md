# README

## Requirements
Follow the project-level requirement with `openai>=1.26.0` and `sglang>=0.1.21`.

## Run
First, add the model api config in `./config/api_config.yaml`. Check the default `gen_answer_config.yaml` and `judge_config.yaml`.

Then serve the model in the backend:
```
screen -S serve
bash ./scripts/serve.sh  # make sure to modify the model name
# then detach from this
```

Run and evaluate with the served model.
```
screen -S run
bash ./scripts/run.sh  # make sure to modify the model name
```
