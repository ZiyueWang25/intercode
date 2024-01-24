import io
import os
import tarfile

from docker.models.containers import Container

from intercode.envs import (
  BashEnv, IntercodeEnv, AGENT_OBS, REWARD, ACTION_EXEC
)
from intercode.envs.swe import install

from typing import Dict, Tuple

SPECIAL_COMMANDS = ("COMMAND", "SUBMIT", "QUIT", "PATCH")

def copy_to_container(container: Container, src: str, dst_dir: str):
    """ src shall be an absolute path """
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w|') as tar, open(src, 'rb') as f:
        info = tar.gettarinfo(fileobj=f)
        info.name = os.path.basename(src)
        tar.addfile(info, f)

    container.put_archive(dst_dir, stream.getvalue())

class SWEEnv(BashEnv):
    """Gym environmnet for SWE-bench"""
    name = "ic_swe"

    def __init__(self, image_name: str, **kwargs):
        IntercodeEnv.__init__(self, image_name, **kwargs)
        self.token = os.environ.get("GITHUB_TOKENS")
        self.installer = install.Installer(self.logger, self.container)
        if self.token is None:
            raise ValueError("'GITHUB_TOKENS' is not specified as environment variable.")

    def reset_container(self) -> None:
        self.workdir = "/"
        folders = self.container.exec_run(self.clean_cmd('ls')).output.decode("utf-8")        

        # Clone repository if not already cloned
        repo_name = self.record['repo'].replace("/", "__")
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
        self.container.exec_run(self.clean_cmd("git reset HEAD ."), workdir=self.workdir)
        self.container.exec_run(self.clean_cmd("git clean -fdx"), workdir=self.workdir)
        self.container.exec_run(
            self.clean_cmd(f"git -c advice.detachedHead=false checkout {self.record['base_commit']}"),
            workdir=self.workdir)

        self.apply_patch(self.record['tests']['patch'], rm=False)

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

        if sum(x in action for x in SPECIAL_COMMANDS) > 1:
            self.observation = f"Your action contain more than 1 special command. Only one of {SPECIAL_COMMANDS} is allowed per action."
            return self.observation, 0, False, self.info

        if "COMMAND" in action:
            if "nano " in action:
                self.observation = "You cannot manually edit the file. You are only allowed to use PATCH with the desired diff."
                return self.observation, 0, False, self.info
            if "rm " in action:
                self.observation = "You cannot remove any file. You are only allowed to use PATCH with the desired diff."
                return self.observation, 0, False, self.info
            self.exec_action(self.extract_command(action))

        if "PATCH" in action:
            patch = self.extract_patch(action)
            self.info["patch"] = patch
            file = patch.split("---")[1].split("+++")[0].split("/")[-1].strip()
            if "test_" in file or "_test.py" in file:
                self.observation = "You cannot edit test file."
                return self.observation, 0, False, self.info

            exit_code, output = self.apply_patch(patch, rm=False)
            self.observation = output.decode("utf-8")
            self.info[ACTION_EXEC] = exit_code == 0

        if "SUBMIT" in action:
            self.exec_action("pytest")
            reward, info = self.get_reward()
            if self.traj_dir is not None:
                self.save_trajectory()
            return self.observation, reward, True, info

        if "SKIP" in action:
            self.trajectory.append((action, ""))
            return "SKIP", 0, True, self.info 
        
            
        self.logger.info(f"Action: {action}")
        self.logger.info(f"Observation: {self.observation}")
        self.trajectory.append((action, self.observation))
        return self.observation, 0, False, self.info

    def apply_patch(self, patch: str, rm=True):
        orig_patch_path = "./patch.diff"
        patch_path = os.path.abspath(orig_patch_path)
        with open(patch_path, "w") as f:
            f.write(patch)
        copy_to_container(self.container, patch_path, self.workdir)
        exit_code, output = self.container.exec_run(
            self.clean_cmd(f"cat {orig_patch_path}"),
            workdir=self.workdir
        )
        if exit_code != 0:
            raise ValueError(f"patch.diff doesn't exist: {output.decode()}")

        # Apply patch to testbed directory
        exec_result = self.container.exec_run(
            self.clean_cmd(f"git apply -v {orig_patch_path}"),
            workdir=self.workdir
        )
        if exec_result.exit_code != 0:
            raise ValueError(f"failed to apply patch: {exec_result}")
        self.logger.info("Successfully applied patch")
        os.remove(patch_path)
        if rm:
            self.container.exec_run(
                self.clean_cmd(f"rm {orig_patch_path}"),
                workdir=self.workdir
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
        reward, info = 1, {}
        if "failed" in self.observation:
            reward = 0
        return reward, info
    
    def close(self):
        self.logger.info("Beginning environment shutdown...")
        self.container.stop()
        self.logger.info("Agent container stopped")