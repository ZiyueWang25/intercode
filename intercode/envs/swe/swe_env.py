import io
import os
import tarfile

from docker.models.containers import Container

from intercode.envs import BashEnv, IntercodeEnv, AGENT_OBS, REWARD, ACTION_EXEC
from intercode.envs.swe import install
from intercode.envs.swe import util

from typing import Dict, Tuple

SPECIAL_COMMANDS = ("COMMAND", "SUBMIT", "SKIP", "PATCH")


class SWEEnv(BashEnv):
    """Gym environmnet for SWE-bench"""

    name = "ic_swe"

    def __init__(self, image_name: str, **kwargs):
        IntercodeEnv.__init__(self, image_name, **kwargs)
        self.token = os.environ.get("GITHUB_TOKENS")
        self.installer = install.Installer(self.logger, self.container)
        if self.token is None:
            raise ValueError(
                "'GITHUB_TOKENS' is not specified as environment variable."
            )

    def reset_container(self) -> None:
        self.workdir = "/"
        folders = self.container.exec_run(self.clean_cmd("ls")).output.decode("utf-8")

        # Clone repository if not already cloned
        repo_name = self.record["repo"].replace("/", "__")
        if repo_name not in folders:
            self.logger.info(f"{repo_name} not found in container, cloning...")
            if "ZiyueWang25" in repo_name:
                clone_cmd = f"git clone https://github.com/ziyuewang25/{repo_name}.git"
            else:
                clone_cmd = f"git clone https://github.com/swe-bench/{repo_name}.git"

            code, output = self.container.exec_run(self.clean_cmd(clone_cmd))
            if code != 0:
                raise ValueError(f"failed to clone repo: {output.decode()}")

        self.installer.install_pkg(self.record)

        # Clean repository of any modifications + Checkout base commit
        self.workdir = f"/{repo_name}/"
        self.container.exec_run(self.clean_cmd("git status"), workdir=self.workdir)
        self.container.exec_run(self.clean_cmd("git restore ."), workdir=self.workdir)
        self.container.exec_run(
            self.clean_cmd("git reset HEAD ."), workdir=self.workdir
        )
        self.container.exec_run(self.clean_cmd("git clean -fdx"), workdir=self.workdir)
        self.container.exec_run(
            self.clean_cmd(
                f"git -c advice.detachedHead=false checkout {self.record['base_commit']}"
            ),
            workdir=self.workdir,
        )

        self.apply_patch(self.record["tests"]["patch"], rm=False)

    def step(self, action: str) -> Tuple[str, int, bool, Dict]:
        """
        Runs given action in environment and returns corresponding output

        Args:
            action (`str`) - command to run in bash shell

        Returns:
            observation (`str`) - standard output
            reward (`float`) - value between 0 and 1 quantifying correctness of output + environment state
            done (`bool`) - whether task is over
            info (`dict`) - additional information (e.g. debugging information)
        """
        if not any(x in action for x in SPECIAL_COMMANDS):
            self.observation = f"Your action doesn't contain {SPECIAL_COMMANDS}"
            self.info[ACTION_EXEC] = False

        if sum(x in action for x in SPECIAL_COMMANDS) > 1:
            self.observation = f"Your action contain more than 1 special command. Only one of {SPECIAL_COMMANDS} is allowed per action."
            self.info[ACTION_EXEC] = False
            return self.observation, 0, False, self.info

        if "COMMAND" in action:
            if "nano " in action:
                self.observation = "You cannot manually edit the file. You are only allowed to use PATCH with the desired diff."
                self.info[ACTION_EXEC] = False
                return self.observation, 0, False, self.info
            if "rm " in action:
                self.observation = "You cannot remove any file. You are only allowed to use PATCH with the desired diff."
                self.info[ACTION_EXEC] = False
                return self.observation, 0, False, self.info
            self.exec_action(self.extract_command(action))

        if "PATCH" in action:
            patch = self.extract_patch(action)
            self.info[ACTION_EXEC] = False
            self.info["patch"] = patch
            try:
                file = patch.split("---")[1].split("+++")[0].split("/")[-1].strip()
            except IndexError:
                self.observation = "The patch format is wrong."
                return self.observation, 0, False, self.info
            if "test_" in file or "_test.py" in file:
                self.observation = "You cannot edit test file."
                return self.observation, 0, False, self.info

            self.apply_patch(patch, rm=False)

        if "SUBMIT" in action:
            self.exec_action("pytest")
            reward = self.get_reward()
            return self.observation, reward, True, self.info

        if "SKIP" in action:
            return super().step("skip")

        return self.observation, 0, False, self.info

    def apply_patch(self, patch: str, rm=True):
        orig_patch_path = "./patch.diff"
        patch_path = os.path.abspath(orig_patch_path)
        with open(patch_path, "w") as f:
            f.write(patch)
        util.copy_to_container(self.container, patch_path, self.workdir)

        # Apply patch to testbed directory
        self.exec_action(f"git apply -v {orig_patch_path}")
        if rm:
            os.remove(patch_path)
            self.container.exec_run(
                self.clean_cmd(f"rm {orig_patch_path}"), workdir=self.workdir
            )

    def extract_command(self, action):
        exec_action = action.split("COMMAND:")[1].lstrip()
        if "`" in exec_action:
            exec_action = action.split("`")[1]
        return exec_action

    def extract_patch(self, action):
        patch = action.split("PATCH:")[1].lstrip()
        if "```" in patch:
            patch = patch.split("```")[1]
        return patch

    def get_reward(self) -> Tuple[float, Dict]:
        return int("failed" not in self.observation)

    def close(self):
        self.logger.info("Beginning environment shutdown...")
        self.container.stop()
        self.logger.info("Agent container stopped")
