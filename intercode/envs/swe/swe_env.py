import io
import os
import tarfile

from docker.models.containers import Container

from intercode.envs import BashEnv, IntercodeEnv, AGENT_OBS, REWARD, ACTION_EXEC
from intercode.envs.swe import install
from intercode.envs.swe import util
from intercode.envs.swe import extract

from typing import Dict, Tuple


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

        self.apply_patch(self.record["tests"]["patch"], rm=True)

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
        self.observation = ""
        reward = 0
        self.info = {ACTION_EXEC: False}
        done = False
        commands = extract.get_commands(action)
        if not commands:
            self.observation += (
                f"Your action doesn't contain {(v for v in extract.SpecialCommandType)}"
            )
        for cnt, command in enumerate(commands):
            self.observation += f"\nCommand {cnt} result:\n"
            if command.type == extract.SpecialCommandType.SUBMIT:
                self.observation += "Submit\n"
                self.exec_action("pytest")
                if self.info[ACTION_EXEC]:
                    reward = int("failed" not in self.observation)
                done = True
            if command.type == extract.SpecialCommandType.SKIP:
                self.observation += "Skip"
                self.info[ACTION_EXEC] = True
                done = True
            if command.type == extract.SpecialCommandType.PATCH:
                self.apply_patch(command.content)
            if command.type == extract.SpecialCommandType.SHELL:
                self.exec_shell(command.content)

            if done:
                break

        return self.observation, reward, done, self.info

    def exec_shell(self, shell_content: str):
        if "nano " in shell_content:
            self.observation += "You cannot manually edit the file. You are only allowed to use PATCH with the desired diff."
            self.info[ACTION_EXEC] = False
            return
        if "rm " in shell_content:
            self.observation += "You cannot remove any file. You are only allowed to use PATCH with the desired diff."
            self.info[ACTION_EXEC] = False
            return
        self.exec_action(shell_content)

    def apply_patch(self, patch: str, rm=True):
        self.info["patch"] = patch
        try:
            file = patch.split("---")[1].split("+++")[0].split("/")[-1].strip()
        except IndexError:
            self.observation += "The patch format is wrong."
            self.info[ACTION_EXEC] = False
            return
        if "test_" in file or "_test.py" in file:
            self.observation += "You cannot edit test file."
            self.info[ACTION_EXEC] = False
            return

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

    def close(self):
        self.logger.info("Beginning environment shutdown...")
        self.container.stop()
        self.logger.info("Agent container stopped")
