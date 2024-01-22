import argparse
import readline

from intercode.envs import (
    BashEnv, PythonEnv, SqlEnv, CTFEnv, SWEEnv
)
from experiments.policies import (
    HumanPolicy, ChatGPTPolicy
)
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
    "bash": {"env": BashEnv, "image_name": "intercode-nl2bash", "data_path": "./data/nl2bash/nl2bash_fs_1.json"},
    "python": {"env": PythonEnv, "image_name": "intercode-python", "data_path": "./data/python/mbpp/ic_mbpp.json"},
    "sql": {"env": SqlEnv, "image_name": "docker-env-sql", "data_path": "./data/sql/bird/ic_bird.json", "preprocess": preprocess_sql},
    "ctf": {"env": CTFEnv, "image_name": "intercode-ctf", "data_path": "./data/ctf/ic_ctf.json", "preprocess": preprocess_ctf},
    "swe": {"env": SWEEnv, "image_name": "intercode-swe", "data_path": "./data/swe-bench/ic_swe_bench_dev.json"}
}


def main(args):
    env = args.env
    if env not in ENV_MAP:
        raise ValueError(f"env {env} not supported (Specify one of [bash, python, sql])")
    image_name = ENV_MAP[env]["image_name"]
    data_path = ENV_MAP[env]["data_path"] if "data_path" in ENV_MAP[env] else None
    preprocess = ENV_MAP[env]["preprocess"] if "preprocess" in ENV_MAP[env] else None

    env = ENV_MAP[env]["env"](image_name, data_path=data_path, verbose=True, preprocess=preprocess, traj_dir="./traj_dir/")
    human_policy = HumanPolicy() if "human" in args.mode else None
    ai_policy = None
    if "ai" in args.mode:
        ai_policy = ChatGPTPolicy(language=LANG_BY_ENV[args.env], setting=SETTING_MAP[args.env],
            template=args.template, dialogue_limit=args.dialogue_limit, model=args.model, verbose=True)

    try:
        for idx in range(min(len(env.data_loader), 3)):
            env.reset(idx)
            if ai_policy is not None:
                ai_policy.reset()
            obs, reward, done = None, None, False
            query = env.query if hasattr(env, "query") else None
            print(f'------\nQuery {idx}: {env.query}')
            turn = 0
            while not done:
                print(f"- Turn {turn}")
                turn += 1
                if args.mode == "human":
                    action = human_policy.forward(query, obs, env.get_available_actions())
                elif args.mode == "ai":
                    action, _ = ai_policy.forward(query, obs, reward, env.get_available_actions())
                elif args.mode == "human_ai":
                    ai_action, _ = ai_policy.forward(query, obs, reward, env.get_available_actions())
                    print(f"-- AI Action: {ai_action}")
                    action = human_policy.forward(query, ai_action, env.get_available_actions())
                    if action == "":
                        action = ai_action
                else:
                    raise ValueError(f"mode {args.mode!r} is not supported")

                print(f"-- Action: {action}")
                obs, reward, done, info = env.step(action)
                if reward == 1:
                    print("Solved!")
                    done = True

    except KeyboardInterrupt:
        print("Exiting InterCode environment...")
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, choices=["bash", "python", "sql", "ctf", "swe"], help="Environment to run [bash, python, sql, ctf, swe]", default="bash")
    parser.add_argument("--mode", type=str, choices=["human", "ai", "human_ai"], help="Mode about the model", default="human_ai")
    parser.add_argument('--model', type=str, help="model to use for AI policy", default="gpt-4-1106-preview")
    parser.add_argument('--dialogue_limit', type=int, help='maximum number of turns in the policy\'s dialogue to keep, only used when the mode is "ai"', default=999)
    parser.add_argument('--max_turns', type=int, help='max number of interaction turns, only used when the mode is "ai"', default=999)
    parser.add_argument('--template', type=str, help="template to use for prompting strategy", default="v2")

    args = parser.parse_args()
    main(args)