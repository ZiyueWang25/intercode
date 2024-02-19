"""Run demo for each environment.

Example usage:
python run_demo.py swe --mode=ai --model=claude --template=swe  --use_toy_example
"""

import argparse
import readline
import os

from intercode.envs import BashEnv, PythonEnv, SqlEnv, CTFEnv, SWEEnv, AGENT_OBSERVATION
from experiments.policies import HumanPolicy, ChatGPTPolicy, ChatAnthropicPolicy
from experiments.logger_helper import Logger
from experiments.utils import HANDICAP_MAP, PROMPT_MAP, SETTING_MAP, LANG_BY_ENV

from typing import Dict, List


def preprocess_ctf(record: Dict) -> List:
    cmds = [f"cd /ctf/{record['task_id']}"]
    if "setup" in record:
        cmds.append(record["setup"])
    return cmds


def preprocess_sql(record: Dict) -> List:
    db = record["db"]
    return [f"use {db}"]


ENV_MAP = {
    "bash": {
        "env": BashEnv,
        "image_name": "intercode-nl2bash",
        "data_path": "./data/nl2bash/nl2bash_fs_1.json",
    },
    "python": {
        "env": PythonEnv,
        "image_name": "intercode-python",
        "data_path": "./data/python/mbpp/ic_mbpp.json",
    },
    "sql": {
        "env": SqlEnv,
        "image_name": "docker-env-sql",
        "data_path": "./data/sql/bird/ic_bird.json",
        "preprocess": preprocess_sql,
    },
    "ctf": {
        "env": CTFEnv,
        "image_name": "intercode-ctf",
        "data_path": "./data/ctf/ic_ctf.json",
        "preprocess": preprocess_ctf,
    },
    "swe": {
        "env": SWEEnv,
        "image_name": "intercode-swe",
        "data_path": "./data/swe-bench/ic_swe_bench_dev_sorted.json",
    },
}


def main(args):
    env = args.env
    if env not in ENV_MAP:
        raise ValueError(
            f"env {env} not supported (Specify one of [bash, python, sql])"
        )
    if env == "swe" and args.use_toy_example:
        print("Use toy example")
        ENV_MAP["swe"]["data_path"] = "./data/swe-bench/ic_toy_examples.json"

    os.makedirs(args.log_dir, exist_ok=True)
    logger = Logger(filename=os.path.join(args.log_dir, "demo"))

    image_name = ENV_MAP[env]["image_name"]
    data_path = ENV_MAP[env]["data_path"] if "data_path" in ENV_MAP[env] else None
    preprocess = ENV_MAP[env]["preprocess"] if "preprocess" in ENV_MAP[env] else None

    env = ENV_MAP[env]["env"](
        image_name,
        data_path=data_path,
        verbose=True,
        preprocess=preprocess,
        traj_dir="./traj_dir/",
        logger=logger,
    )
    human_policy = HumanPolicy() if "human" in args.mode else None
    ai_policy = None
    if "ai" in args.mode:
        policy_args = dict(
            language=LANG_BY_ENV[args.env],
            setting=SETTING_MAP[args.env],
            template=args.template,
            dialogue_limit=args.dialogue_limit,
            model=args.model,
        )

        if args.model.lower() == "claude":
            ai_policy = ChatAnthropicPolicy(**policy_args)
        else:
            ai_policy = ChatGPTPolicy(**policy_args)
        logger.debug(f"Dialogue Controller: {ai_policy.dialogue_controller}")

    try:
        for idx in range(len(env.data_loader)):
            env.reset(idx)
            record = env.data_loader.get(idx)
            logger.msg_record(record)
            logger.log_episode(env, record, idx)
            if ai_policy is not None:
                ai_policy.reset()
            observation, reward, done = None, None, False
            query = env.query if hasattr(env, "query") else None
            logger.info(f"System Message:\n {ai_policy.template.get_init_msg()}")
            logger.info(f"Query Message:\n {ai_policy.template.get_query_msg(query)}")
            turn = 0
            while not done:
                turn += 1
                if args.mode == "human":
                    action = human_policy.forward(query, observation)
                elif args.mode == "ai":
                    action, _ = ai_policy.forward(query, observation, reward)
                elif args.mode == "human_ai":
                    ai_action, _ = ai_policy.forward(query, observation, reward)
                    logger.info(f"-- AI Action: {ai_action}")
                    action = human_policy.forward(query, ai_action)
                    if action == "":
                        action = ai_action
                else:
                    raise ValueError(f"mode {args.mode!r} is not supported")
                observation, reward, done, info = env.step(action)
                info[AGENT_OBSERVATION] = ai_policy.dialogue_controller[-2]["content"]
                logger.msg_turn(turn, observation, action, reward, done, info)
                logger.log_turn_history(idx, str(observation), action, reward, info)
                if reward == 1:
                    logger.info("Solved!")
                    done = True
                if turn > args.max_turns:
                    logger.info("Exceed max turn, Skip")
                    done = True
            logger.log_summary(idx)
            logger.info(f"Query {idx} Finished")

    except KeyboardInterrupt:
        logger.info("Exiting InterCode environment...")
    finally:
        logger.save_turn()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env",
        type=str,
        choices=["bash", "python", "sql", "ctf", "swe"],
        help="Environment to run [bash, python, sql, ctf, swe]",
        default="bash",
    )
    parser.add_argument(
        "--use_toy_example",
        action="store_true",
        help="use toy example in swe environment",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["human", "ai", "human_ai"],
        help="Mode about the model",
        default="human_ai",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["claude", "gpt-4-1106-preview", "gpt-3.5-turbo"],
        help="model to use for AI policy",
        default="gpt-4-1106-preview",
    )
    parser.add_argument(
        "--dialogue_limit",
        type=int,
        help='maximum number of turns in the policy\'s dialogue to keep, only used when the mode is "ai"',
        default=40,
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        help='max number of interaction turns, only used when the mode is "ai"',
        default=20,
    )
    parser.add_argument(
        "--template",
        type=str,
        help="template to use for prompting strategy",
        default="v2",
    )
    parser.add_argument(
        "--log_dir", type=str, help="directory to save log", default="./logs/demo/"
    )

    args = parser.parse_args()
    main(args)
