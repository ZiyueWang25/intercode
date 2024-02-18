import os
import datetime
import json
import logging
import sys
from typing import Dict


def add_time_suffix(path):
    suffix = datetime.datetime.now().strftime("%y%m%d-%H:%M:%S")
    return add_suffix(path, suffix)


def add_suffix(path, suffix):
    if not path:
        return ""
    path_wo_ext, ext = os.path.splitext(path)
    return path_wo_ext + "_" + suffix + ext


def get_msg_logger(
    filename: str = "", file_level=logging.INFO, stdout_level=logging.DEBUG
):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(stdout_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if filename:
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TurnLogger:
    def __init__(self, filename: str):
        self.log_path = filename
        self.log_data = {}

    def _init_turn_history(self):
        return {"action": [], "observation": [], "reward": [], "info": []}

    def log_turn_history(
        self,
        idx: int,
        observation: str = "",
        action: str = "",
        reward: float = 0.0,
        info: bool = False,
    ):
        if idx not in self.log_data:
            raise ValueError("turn history is not initialized")
        self.log_data[idx]["turn_history"]["observation"].append(observation)
        self.log_data[idx]["turn_history"]["action"].append(action)
        self.log_data[idx]["turn_history"]["reward"].append(reward)
        self.log_data[idx]["turn_history"]["info"].append(
            {
                str(k): [str(x) for x in v] if isinstance(v, list) else str(v)
                for k, v in info
            }
        )

    def log_episode(self, env, record: Dict, idx: int):
        log_episode = {
            "environment": env.name,
            "dataset": env.data_path,
            "task_id": idx,
            "query": env.query,
            "gold": env.gold,
            "turn_history": self._init_turn_history(),
        }
        if "hardness" in record:
            log_episode["hardness"] = record["hardness"]
        self.log_data[idx] = log_episode

    def log_summary(self, idx: int):
        if "summary" in self.log_data[idx]:
            return
        turn_history = self.log_data[idx]["turn_history"]
        reward = 0 if not turn_history["reward"] else turn_history["reward"][-1]
        self.log_data[idx]["summary"] = {
            "final_reward": reward,
            "turns_taken": len(turn_history) + 1,
        }

    def save_turn(self):
        with open(self.log_path, "w") as fp:
            json.dump(self.log_data, fp, indent=2)


class Logger:
    def __init__(
        self, filename: str = "", file_level=logging.DEBUG, stdout_level=logging.INFO
    ):
        filename = add_time_suffix(filename)
        msg_file = add_suffix(filename, "msg") + ".txt" if filename else ""
        self.msg_logger = get_msg_logger(msg_file, file_level, stdout_level)
        turn_file = add_suffix(filename, "turn") + ".json" if filename else ""
        self.turn_logger = TurnLogger(turn_file)
        self.disabled = False

    def info(self, msg: str):
        self.msg_logger.disabled = self.disabled
        self.msg_logger.info(msg)

    def debug(self, msg: str):
        self.msg_logger.disabled = self.disabled
        self.msg_logger.debug(msg)

    def warning(self, msg: str):
        self.msg_logger.disabled = self.disabled
        self.msg_logger.warning(msg)

    def error(self, msg: str):
        self.msg_logger.disabled = self.disabled
        self.msg_logger.error(msg)

    def msg_record(self, record: Dict):
        self.msg_logger.info("Record")
        for key in ["repo", "version", "task_id"]:
            self.msg_logger.info(f"{key}: {record[key]}")

    def msg_turn(self, turn, observation, action, reward, done, info):
        self.info("#" * 20 + f" Turn {turn} " + "#" * 20)
        self.info(f"-- ACTION:\n{action}")
        if len(observation) > 200:
            self.info(f"-- OBSERVATION:\n{observation[:200]} ...")
        else:
            self.info(f"-- OBSERVATION:\n{observation}")

        self.info(f"-- REWARD:\n{reward}")
        self.info(f"-- DONE:\n{done}")
        self.debug(f"-- INFO:\n{info}")

    def log_turn_history(
        self,
        idx: int,
        obs: str = "",
        act: str = "",
        reward: float = 0.0,
        info: dict = {},
    ):
        self.turn_logger.log_turn_history(idx, obs, act, reward, info)

    def log_episode(self, env, record: Dict, idx: int):
        self.turn_logger.log_episode(env, record, idx)

    def log_summary(self, idx: int):
        self.turn_logger.log_summary(idx)

    def save_turn(self):
        self.turn_logger.save_turn()
