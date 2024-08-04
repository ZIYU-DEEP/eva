"""
Usage:
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode single
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import dataclasses

import numpy as np
from tqdm import tqdm

from common import (
    load_questions,
    load_model_answers,
    load_judge_prompts,
    check_data,
    play_a_match_single,
    get_model_list,
    Judge,
    MatchSingle,
    NEED_REF_CATS,
)

# Function to load existing judgments from a file if it exists
def load_existing_judgments(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            existing_judgments = set(line.strip() for line in f)
    else:
        existing_judgments = set()
    return existing_judgments

# Function to create matches for single answer grading
def make_match_single(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m = models[i]
            a = model_answers[m][q_id]
            if ref_answers is not None:
                ref = ref_answers[judge.model_name][q_id]
                matches.append(
                    MatchSingle(
                        dict(q), m, a, judge, ref_answer=ref, multi_turn=multi_turn
                    )
                )
            else:
                matches.append(MatchSingle(dict(q), m, a, judge, multi_turn=multi_turn))
    return matches

# Function to create a Judge object for single answer grading
def make_judge_single(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["single-v1"])
    judges["math"] = Judge(judge_model, judge_prompts["single-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(
        judge_model, judge_prompts["single-v1-multi-turn"], multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["single-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name", type=str, default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--judge-file", type=str, default="data/judge_prompts.jsonl",
        help="The file of judge prompts.",
    )
    parser.add_argument(
        "--judge-model", type=str, default="gpt-4"
    )
    parser.add_argument(
        "--baseline-model", type=str, default="gpt-3.5-turbo"
    )
    parser.add_argument(
        "--mode", type=str, default="single",
        choices=["single"],
        help=(
            "Evaluation mode. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument(
        "--model-list", type=str, nargs="+", default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, 
        help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--first-n", type=int, 
        help="A debug option. Only run the first `n` judgments."
    )
    parser.add_argument(
        "--filename-suffix", "-fs",  type=str, 
        help="Suffix for the filename",
        default="",
    )
    args = parser.parse_args()

    # Set the paths
    question_file = f"data/{args.bench_name}/question.jsonl"
    answer_dir = f"data/{args.bench_name}/model_answer"
    ref_answer_dir = f"data/{args.bench_name}/reference_answer"

    # Load questions
    questions = load_questions(question_file, None, None)

    # Load answers
    model_answers = load_model_answers(answer_dir)
    ref_answers = load_model_answers(ref_answer_dir)

    # Load judge
    judge_prompts = load_judge_prompts(args.judge_file)

    # DEBUG: only run the first n questions
    if args.first_n:
        questions = questions[: args.first_n]

    if args.model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = args.model_list

    if args.mode == "single":
        judges = make_judge_single(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_single

        combined_matches = []
        for model_id in models:
            output_file = (
                f"data/{args.bench_name}/model_judgment/{model_id}_single.jsonl"
            )

            # Skip the judgment call if the model file already exists
            if os.path.exists(output_file):
                print(f"Skipping {model_id} as judgments already exist.")
                continue

            question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
            question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

            matches = []
            matches += make_match_single(
                question_default, [model_id], model_answers, judges["default"], None, None, False
            )
            matches += make_match_single(
                question_math, [model_id], model_answers, judges["math"], None, ref_answers, False
            )
            matches += make_match_single(
                question_default, [model_id], model_answers, judges["default-mt"], None, None, True
            )
            matches += make_match_single(
                question_math, [model_id], model_answers, judges["math-mt"], None, ref_answers, True
            )

            match_stat = {}
            match_stat["bench_name"] = args.bench_name
            match_stat["mode"] = args.mode
            match_stat["judge"] = args.judge_model
            match_stat["baseline"] = None
            match_stat["model_id"] = model_id
            match_stat["total_num_questions"] = len(questions)
            match_stat["total_num_matches"] = len(matches)
            match_stat["output_path"] = output_file

            # Show match stats and prompt enter to continue
            print("Stats for model:", model_id)
            print(json.dumps(match_stat, indent=4))
            input("Press Enter to confirm...")

            # Play matches
            if args.parallel == 1:
                for match in tqdm(matches):
                    play_a_match_func(match, output_file=output_file)
            else:
                def play_a_match_wrapper(match):
                    play_a_match_func(match, output_file=output_file)

                np.random.seed(0)
                np.random.shuffle(matches)

                with ThreadPoolExecutor(args.parallel) as executor:
                    for match in tqdm(
                        executor.map(play_a_match_wrapper, matches), total=len(matches)
                    ):
                        pass

            combined_matches.extend(matches)

        # Combine all judgment files into a single file
        combined_output_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_single_combined{args.filename_suffix}.jsonl"
        )
        with open(combined_output_file, 'w') as combined_f:
            for model_id in models:
                model_file = (
                    f"data/{args.bench_name}/model_judgment/{model_id}_single.jsonl"
                )
                if os.path.exists(model_file):
                    with open(model_file, 'r') as f:
                        for line in f:
                            combined_f.write(line)
    else:
        raise NotImplementedError
