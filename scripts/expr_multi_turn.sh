# !bin/bash
set -x
set -e

# Data Paths
# - (SQL)  ./data/sql/spider/ic_spider_dev.json
# - (SQL)  ./data/test/sql_queries.csv
# - (Bash) ./data/nl2bash/nl2bash.json
# - (Bash) ./data/test/bash_queries.json

# Environments
# - sql, bash

# Image Names
# - (SQL)  docker-env-sql
# - (Bash) intercode-bash
# - (Bash) intercode-nl2bash
# - (Py)   intercode-python

# Policies
# - chat

# Bash Call
# python -m experiments.eval_n_turn \
#     --data_path ./data/nl2bash/nl2bash_fs_1.json \
#     --dialogue_limit 7 \
#     --env bash \
#     --image_name intercode-nl2bash \
#     --log_dir logs/experiments \
#     --max_turns 10 \
#     --policy_type chat \
#     --template v2 \
#     --model gpt-3.5-turbo \
#     --verbose

# SQL Call
# python -m experiments.eval_n_turn \
#     --data_path ./data/sql/spider/ic_spider_dev.json \
#     --dialogue_limit 5 \
#     --env sql \
#     --image_name docker-env-sql \
#     --log_dir logs/experiments \
#     --max_turns 10 \
#     --policy_type chat \
#     --template game_sql \
#     --model gpt-3.5-turbo
#     --handicap 
#     --verbose 

# Python call
# python -m experiments.eval_n_turn \
#     --data_path ./data/python/mbpp/ic_mbpp.json \
#     --dialogue_limit 5 \
#     --env python \
#     --image_name intercode-python \
#     --log_dir logs/experiments \
#     --max_turns 10 \
#     --policy_type chat \
#     --template function \
#     --model gpt-3.5-turbo \
#     --verbose

# CTF call
# python -m experiments.eval_n_turn \
#     --data_path ./data/ctf/ic_ctf.json \
#     --dialogue_limit 5 \
#     --env ctf \
#     --image_name intercode-ctf \
#     --log_dir logs/experiments \
#     --max_turns 10 \
#     --policy_type chat \
#     --template ctf \
#     --model gpt-4 \
#     --verbose

# SWE Call
python -m experiments.eval_n_turn \
    --data_path ./data/swe-bench/ic_swe_bench_dev_sorted.json \
    --dialogue_limit 40 \
    --env swe \
    --image_name intercode-swe \
    --log_dir logs/experiments \
    --max_turns 20 \
    --policy_type chat \
    --template swe \
    --model claude \
    --num_tasks -1 \
    --verbose