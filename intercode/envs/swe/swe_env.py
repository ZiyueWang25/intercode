import os
from intercode.envs import BashEnv, IntercodeEnv, EXEC_RESULTS
from intercode.envs.swe import install
from intercode.envs.swe import util
from intercode.envs.swe import extract
from intercode.envs.exec_result import ExecResult, SkipResult

from typing import Dict, Tuple

# import io
# import tarfile
# from docker.models.containers import Container  # type: ignore
# from intercode.envs import AGENT_OBS, REWARD


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
            user = "ziyuewang" if "ZiyueWang25" in repo_name else "swe-bench"
            clone_cmd = f"git clone https://github.com/{user}/{repo_name}.git"
            self.logger.debug(f"Clone: {clone_cmd}")
            is_valid = self.container.exec_run(clone_cmd)
            if not is_valid:
                raise ValueError(
                    f"failed to clone repo: {self.info[EXEC_RESULTS][-1].output}"
                )

        self.installer.install_pkg(self.record)

        # Clean repository of any modifications + Checkout base commit
        self.workdir = f"/{repo_name}/"
        reset_commands = [
            "git fetch",
            f"git reset --hard {self.record['base_commit']}",
            "git clean -fdx",
            f"git -c advice.detachedHead=false checkout {self.record['base_commit']}",
        ]
        for c in reset_commands:
            if not self.container.exec_run(c):
                raise RuntimeError(
                    f"failed to execute {c!r}: {self.info[EXEC_RESULTS][-1].output}"
                )

        if not self.apply_patch(
            self.record["tests"]["patch"], rm=True, allow_test_edit=True
        ):
            raise RuntimeError(
                f"failed to apply gold test patch: {self.info[EXEC_RESULTS][-1].output}"
            )

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
        self.info = {EXEC_RESULTS: []}
        done = False
        commands = extract.get_commands(action)
        if not commands:
            warn = f"Your action doesn't contain {tuple(v.value for v in extract.SpecialCommandType)}"
            self.info[EXEC_RESULTS].append(ExecResult(action, 1, warn))
        for command in commands:
            if command.type == extract.SpecialCommandType.SUBMIT:
                self.exec_action("echo 'Submit' && pytest")
                if self.info[EXEC_RESULTS][-1].is_valid:
                    reward = int("failed" not in self.info[EXEC_RESULTS][-1].output)
                done = True
            if command.type == extract.SpecialCommandType.SKIP:
                self.info[EXEC_RESULTS].append(SkipResult(action))
                done = True
            if command.type == extract.SpecialCommandType.PATCH:
                self.apply_patch(command.content)
            if command.type == extract.SpecialCommandType.SHELL:
                self.exec_shell(command.content)

            if done or not self.info[EXEC_RESULTS][-1].is_valid:
                break
        if len(self.info[EXEC_RESULTS]) > 1:
            self.observation = "\n".join(
                [
                    f"Command {i} result: {exec_result.output}"
                    for i, exec_result in enumerate(self.info[EXEC_RESULTS])
                ]
            )
        elif len(self.info[EXEC_RESULTS]) == 1:
            self.observation = self.info[EXEC_RESULTS][0].output
        return self.observation, reward, done, self.info

    def exec_shell(self, shell_content: str):
        path_conda = os.path.abspath("/miniconda3")
        path_activate = os.path.join(path_conda, "bin", "activate")
        repo = self.record["repo"]
        version = self.record["version"]
        repo_prefix = repo.replace("/", "__")
        env_name = f"{repo_prefix}__{version}"
        if "nano " in shell_content:
            warn = "You cannot manually edit the file. You are only allowed to use PATCH with the desired diff."
            self.info[EXEC_RESULTS].append(ExecResult(shell_content, 1, warn))
            return
        if "rm " in shell_content:
            warn = "You cannot remove any file. You are only allowed to use PATCH with the desired diff."
            self.info[EXEC_RESULTS].append(ExecResult(shell_content, 1, warn))
            return
        self.exec_action(f"source {path_activate} {env_name} && {shell_content}")

    def apply_patch(self, patch: str, rm=True, allow_test_edit=False) -> bool:
        try:
            file = patch.split("---")[1].split("+++")[0].split("/")[-1].strip()
        except IndexError:
            warn = "The patch format is wrong."
            self.info[EXEC_RESULTS].append(ExecResult(patch, 1, warn))
            return
        if not allow_test_edit and ("test_" in file or "_test.py" in file):
            warn = "You cannot edit test file."
            self.info[EXEC_RESULTS].append(ExecResult(patch, 1, warn))
            return

        orig_patch_path = "./patch.diff"
        patch_path = os.path.abspath(orig_patch_path)
        with open(patch_path, "w") as f:
            f.write(patch)
        util.copy_to_container(self.container, patch_path, self.workdir)

        # Apply patch to testbed directory
        is_valid = self.container.exec_run(f"git apply -v {orig_patch_path}")
        if rm:
            os.remove(patch_path)
            self.container.exec_run(f"rm {orig_patch_path}", workdir=self.workdir)
        return is_valid

    def close(self):
        self.logger.info("Beginning environment shutdown...")
        self.container.stop()
        self.logger.info("Agent container stopped")
