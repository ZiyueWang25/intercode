"""Helps the agent to navigate through the repository and files"""

import os

from docker.models.containers import Container


class Navigator:
    def __init__(self, container: Container, repo_name: str, line_limit: int = 100):
        self.container = container
        self.workdir = f"/{repo_name}/"
        self.line_limit = line_limit

    def get_directory_tree(self, dir: str, depth: int = -1) -> str:
        """Returns the tree format in str starting from the current directory.

        For example:
        >> navigator.get_directory("/intercode/", depth=1)
        '''File Tree (depth=1):
        assets/
        data/
        docker/
        ...'''

        >> navigator.get_directory("/intercode/intercode/envs/", level=-1)
        '''File Tree (depth=-1):
        __pycache__/
          __init__.cpython-39.pyc
          ic_env.cpython-39.pyc
        bash/
                __pycache__/
                        ...
                bash_env.py
        ctf/
                ...
        ...'''

        Args:
                dir: The directory to start the tree from.
                depth: The depth of tree to go through. If -1, then all depth.
        """
        dir = self._get_relative_path(dir)
        result = f"Tree Format (depth={depth}):\n"
        return result

    def _get_relative_path(self, path: str) -> str:
        """Returns the relative path to the repository root directory"""
        if not path.startswith(self.workdir):
            raise ValueError(f"path {path!r} doesn't start with {self.workdir!r}")
        return "./" + path[len(self.workdir) :]
