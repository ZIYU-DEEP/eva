#!/bin/bash
set -e


python gen_answer.py 

python gen_judgment.py 

python show_result.py

