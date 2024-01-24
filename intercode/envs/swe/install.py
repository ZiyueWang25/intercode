import os
import requests
import re

from docker.models.containers import Container

from intercode.envs.swe import constants


class Installer:
    def __init__(self, logger, container:Container):
        self.path_conda = os.path.abspath("/miniconda3")
        self.conda_bin_path = os.path.join(self.path_conda, "bin")
        self.path_activate = os.path.join(self.conda_bin_path, "activate")
        self.conda_exec_cmd = os.path.join(self.conda_bin_path, "conda")
        self.logger = logger
        self.container = container

        self.env_list = self.get_conda_env_names()
    
    def get_conda_env_names(self) -> list:
        # Get list of conda environments
        code, output = self.container.exec_run(self.clean_cmd(f"{self.conda_exec_cmd} env list"))
        output = output.decode()
        if code != 0:
            raise ValueError(f"failed to get conda env list: {output}")
        lines = output.split("\n")
        # Store environment names to list
        env_names = []
        for line in lines:
            if line.startswith("#"):
                continue
            if line.strip() == "":
                continue
            parts = line.split()
            if len(parts) == 2:
                env_name = parts[0]
            elif len(parts) == 1:
                env_name = parts[0].split('/')[-1]
            env_names.append(env_name)
        return env_names

    def clean_cmd(self, action: str) -> str:
        """Cleans action string"""
        entrypoint = "/bin/bash"
        if '"' in action:
            self.logger.warning(f'" in action: {action}. You should update it to use \' ')
        return f"{entrypoint} -c \"{action.strip()}\""

    def activate_conda(self, env_name):
        code, output = self.container.exec_run(self.clean_cmd(f". {self.path_activate} {env_name}"))
        if code != 0:
            raise ValueError(f"failed to activate: {output.decode()}")
        self.container.exec_run(self.clean_cmd(f"conda deactivate"))
        if code != 0:
            raise ValueError(f"failed to deactivate base: {output.decode()}")
        self.container.exec_run(self.clean_cmd(f"conda activate {env_name}"))
        if code != 0:
            raise ValueError(f"failed to activate env: {output.decode()}")


    def install_pkg(self, instance):
        repo = instance["repo"]
        version = instance["version"]

        repo_prefix = repo.replace("/", "__")
        # Create conda environment per version of the repo
        install = constants.MAP_VERSION_TO_INSTALL[repo][str(version)]
        # Name for both environment and github repo
        env_name = f"{repo_prefix}__{version}"
        if env_name in self.env_list:
            self.logger.info(f"Conda env: {env_name} exists, activate it")
            self.activate_conda(env_name)
            self.logger.info(f"Activate {env_name} successfully")
            return

        # Create conda environment according to install instructinos
        pkgs = install["packages"] if "packages" in install else ""
        if pkgs == "requirements.txt":
            # Create environment
            code, output = self.container.exec_run(self.clean_cmd(f"{self.conda_exec_cmd} create -n {env_name} python={install['python']} -y"))
            if code != 0:
                raise ValueError(f"failed to install conda: {output.decode()}")
            self.activate_conda(env_name)


            # Install dependencies
            path_to_reqs = get_requirements(instance, self.testbed)
            cmd = f"pip install -r {path_to_reqs}"
            code, output = self.container.exec_run(self.clean_cmd(cmd))
            if code != 0:
                raise ValueError(f"failed to install requirements: {output.decode()}")
            os.remove(path_to_reqs)
        elif pkgs == "environment.yml":
            # Create environment from yml
            path_to_reqs = get_environment_yml(
                instance, env_name, self.testbed
            )
            if "no_use_env" in install and install["no_use_env"]:
                # `conda create` based installation
                cmd = f"{self.conda_exec_cmd} create -c conda-forge -n {env_name} python={install['python']} -y"
                self.logger.info(
                    f"[Testbed] Creating environment {env_name}; Command: {cmd}"
                )
                code, output = self.container.exec_run(self.clean_cmd(cmd))
                if code != 0:
                    raise ValueError(f"failed to create environment: {output.decode()}")                    

                # Install dependencies
                cmd = f"{self.conda_exec_cmd} env update -f {path_to_reqs}"
                self.logger.info(
                    f"[Testbed] Installing dependencies for {env_name}; Command: {cmd}"
                )
                code, output = self.container.exec_run(self.clean_cmd(cmd))
                if code != 0:
                    raise ValueError(f"failed to install dependencies: {output.decode()}")
            else:
                # `conda env create` based installation
                cmd = f"{self.conda_exec_cmd} env create --file {path_to_reqs}"
                self.logger.info(
                    f"[Testbed] Creating environment {env_name}; Command: {cmd}"
                )
                code, output = self.container.exec_run(self.clean_cmd(cmd))
                if code != 0:
                    raise ValueError(f"failed to create environment: {output.decode()}")

            # Remove environment.yml
            os.remove(path_to_reqs)
        else:
            # Create environment + install dependencies
            cmd = f"{self.conda_exec_cmd} create -n {env_name} python={install['python']} {pkgs} -y"
            self.logger.info(
                f"[Testbed] Creating environment {env_name}; Command: {cmd}"
            )
            code, output = self.container.exec_run(self.clean_cmd(cmd))
            if code != 0:
                raise ValueError(f"failed to create environment: {output.decode()}")

        # Install additional packages if specified
        if "pip_packages" in install:
            cmd = f"source {self.path_activate} {env_name} && pip install {install['pip_packages']}"
            self.logger.info(
                f"[Testbed] Installing pip packages for {env_name}; Command: {cmd}"
            )
            code, output = self.container.exec_run(self.clean_cmd(cmd))
            if code != 0:
                raise ValueError(f"failed to install pip packages: {output.decode()}")

def get_requirements(instance: dict, save_path: str = None):
    """
    Get requirements.txt for given task instance

    Args:
        instance (dict): task instance
        save_path (str): If provided, save requirements.txt to this path
    Returns:
        requirements.txt (str): If save_path given, returns path to saved requirements.txt.
            Otherwise, returns requirements.txt as string
    """
    # Attempt to find requirements.txt at each path based on task instance's repo
    path_worked = False
    commit = 'environment_setup_commit' if 'environment_setup_commit' in instance else 'base_commit'

    for req_path in constants.MAP_REPO_TO_REQS_PATHS[instance["repo"]]:
        reqs_url = os.path.join(
            constants.SWE_BENCH_URL_RAW, instance["repo"], instance[commit], req_path
        )
        reqs = requests.get(reqs_url)
        if reqs.status_code == 200:
            path_worked = True
            break
    if not path_worked:
        print(
            f"Could not find requirements.txt at paths {constants.MAP_REPO_TO_REQS_PATHS[instance['repo']]}"
        )
        return None

    lines = reqs.text
    original_req = []
    additional_reqs = []
    req_dir = "/".join(req_path.split("/")[:-1])
    exclude_line = lambda line: any(
        [line.strip().startswith(x) for x in ["-e .", "#", ".[test"]]
    )

    for line in lines.split("\n"):
        if line.strip().startswith("-r"):
            # Handle recursive requirements
            file_name = line[len("-r") :].strip()
            reqs_url = os.path.join(
                constants.SWE_BENCH_URL_RAW,
                instance["repo"],
                instance[commit],
                req_dir,
                file_name,
            )
            reqs = requests.get(reqs_url)
            if reqs.status_code == 200:
                for line_extra in reqs.text.split("\n"):
                    if not exclude_line(line_extra):
                        additional_reqs.append(line_extra)
        else:
            if not exclude_line(line):
                original_req.append(line)

    # Combine all requirements into single text body
    additional_reqs.append("\n".join(original_req))
    all_reqs = "\n".join(additional_reqs)

    if save_path is None:
        return all_reqs

    path_to_reqs = os.path.join(save_path, "requirements.txt")
    with open(path_to_reqs, "w") as f:
        f.write(all_reqs)
    return path_to_reqs

def get_environment_yml(instance: dict, env_name: str, save_path: str = None) -> str:
    """
    Get environment.yml for given task instance

    Args:
        instance (dict): SWE Bench Task instance
        env_name (str): Rename retrieved environment.yml to this name
        save_path (str): If provided, save environment.yml to this path
    Returns:
        environment.yml (str): If save_path given, returns path to saved environment.yml.
            Otherwise, returns environment.yml as string
    """
    # Attempt to find environment.yml at each path based on task instance's repo
    path_worked = False

    commit = 'environment_setup_commit' if 'environment_setup_commit' in instance else 'base_commit'
    for req_path in constants.MAP_REPO_TO_ENV_YML_PATHS[instance["repo"]]:
        reqs_url = os.path.join(
            constants.SWE_BENCH_URL_RAW, instance["repo"], instance[commit], req_path
        )
        reqs = requests.get(reqs_url)
        if reqs.status_code == 200:
            path_worked = True
            break
    if not path_worked:
        print(
            f"Could not find environment.yml at paths {constants.MAP_REPO_TO_ENV_YML_PATHS[instance['repo']]}"
        )
        return None

    lines = reqs.text.split("\n")
    cleaned = []
    for line in lines:
        # Rename environment to given name
        if line.startswith("name:"):
            cleaned.append(f"name: {env_name}")
            continue
        cleaned.append(line)

    # Return environment.yml as string if no save path given
    if save_path is None:
        return "\n".join(cleaned)

    # Save environment.yml to given path and return path
    path_to_reqs = os.path.join(save_path, "environment.yml")
    with open(path_to_reqs, "w") as f:
        f.write("\n".join(cleaned))
    return path_to_reqs

