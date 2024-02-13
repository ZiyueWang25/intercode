import logging
import argparse, os
from tqdm import tqdm

from intercode.envs import ACTION_EXEC, initialize_env
from experiments.policies import initialize_policy
from experiments.logger_helper import Logger
from experiments.utils import HANDICAP_MAP, PROMPT_MAP, SETTING_MAP

parser = argparse.ArgumentParser(
    description="N-turn evaluation for Intercode environment"
)
parser.add_argument(
    "--data_path", type=str, help="path to dataset to evaluate on", required=True
)
parser.add_argument(
    "--dialogue_limit",
    type=int,
    help="maximum number of turns in the policy's dialogue to keep",
    required=True,
)
parser.add_argument(
    "--env",
    choices=["sql", "bash", "python", "ctf", "swe"],
    help="Intercode environment to run eval on",
    required=True,
)
parser.add_argument("--handicap", action="store_true", help="enable handicap")
parser.add_argument(
    "--image_name",
    type=str,
    help="name of docker image to build environment with",
    required=True,
)
parser.add_argument(
    "--log_dir",
    type=str,
    help="folder to save experiment run log file to",
    required=True,
)
parser.add_argument(
    "--max_turns", type=int, help="max number of interaction turns", required=True
)
parser.add_argument(
    "--policy_type",
    choices=["chat", "complete"],
    help="policy type to use for evaluation",
    required=True,
)
parser.add_argument(
    "--template", type=str, help="template to use for prompting strategy", required=True
)
parser.add_argument("--verbose", action="store_true", help="print out logs")
parser.add_argument("--model", type=str, help="model to use for policy", required=True)
parser.add_argument(
    "--num_tasks",
    type=int,
    help="# of tasks to test, if -1, then test all tasks",
    default=20,
)
args = parser.parse_args()


class ExperimentWrapper:
    def __init__(self, args):
        self.args = args

        # Define log file name and path
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        log_file_name = f"{args.env}_multiturn_{args.model}_{args.max_turns}_turns"
        log_path = os.path.join(args.log_dir, log_file_name)
        self.logger = Logger(
            log_path, file_level=logging.INFO, stdout_level=logging.DEBUG
        )
        if not args.verbose:
            self.logger.disabled = True

        # Set environment (No logging for env)
        self.env = initialize_env(
            args.env,
            image_name=args.image_name,
            data_path=args.data_path,
            logger=self.logger,
            verbose=args.verbose,
        )

        # Initialize Policy
        if args.template not in PROMPT_MAP:
            raise ValueError(
                f"Prompt {args.template} not recognized; Options: {PROMPT_MAP.keys()}"
            )
        self.policy = initialize_policy(
            args.policy_type,
            args.model,
            language=args.env,
            setting=SETTING_MAP[args.env],
            template=args.template,
            dialogue_limit=args.dialogue_limit,
        )

        # Initialize handicap
        self.handicap = None
        if args.handicap and args.env in HANDICAP_MAP:
            self.handicap = HANDICAP_MAP[args.env]

    def run_expr(self):
        self.logger.debug("Start experiments")
        try:
            for idx in tqdm(
                range(0, min(len(self.env.data_loader), args.num_tasks)),
                disable=self.args.verbose,
            ):
                self.logger.info("#" * 20 + f" Query {idx} " + "#" * 20)
                # Reset variables per task
                self.env.reset(idx)
                self.policy.reset()
                record = self.env.data_loader.get(idx)
                self.logger.msg_record(record)
                self.logger.log_episode(self.env, record, idx)

                # Add Handicap
                if self.handicap is not None:
                    self.policy.add_to_dialogue(self.handicap(record))

                observation, reward = None, None
                for turn in range(self.args.max_turns):
                    try:
                        action, _ = self.policy.forward(
                            self.env.query, observation, reward
                        )
                    except (ValueError, TypeError) as e:
                        self.logger.error(f"Index {idx}: {e}")
                        self.logger.log_turn_history(action=f"blocked: {e}", reward=0)
                        break

                    observation, reward, done, info = self.env.step(action)

                    self.logger.msg_turn(turn, observation, action, reward, done, info)
                    self.logger.log_turn_history(
                        idx, str(observation), action, reward, info[ACTION_EXEC]
                    )

                    if done:
                        break
                self.logger.log_summary(idx)
                self.logger.info(f"Query {idx} Finished")
        except KeyboardInterrupt:
            print("Keyboard interrupt detected")
        finally:
            self.logger.log_summary(idx)
            self.logger.save_turn()
            self.env.close()


if __name__ == "__main__":
    expr_wrapper = ExperimentWrapper(args)
    expr_wrapper.run_expr()
