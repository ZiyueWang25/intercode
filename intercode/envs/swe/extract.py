"""Extracts information from the agent output"""

import re
from enum import Enum


class SpecialCommandType(Enum):
    SHELL = "SHELL"
    PATCH = "PATCH"
    SKIP = "SKIP"
    SUBMIT = "SUBMIT"


class Command:
    def __init__(self, type: SpecialCommandType, content: str = "", file_path=""):
        self.type = type
        self.content = content
        # file_path is only applicable when the command is about patching.
        self.file_path = file_path


def _extract_shell_or_patch(content: str):
    if "```" in content:
        # obvious code block -- extract it
        return content.split("```")[1]
    if "\n" in content and content[: content.index("\n")].count("`") == 2:
        # obvious one line code block -- extract it
        return content.split("`")[1]
    # extract shell from the single line
    return content.strip().split("\n")[0]


def get_commands(action: str) -> list[Command]:
    pattern = f"({'|'.join([e.value for e in SpecialCommandType])})"
    commands: list[Command] = []
    match_objects = list(re.finditer(pattern, action))
    if not match_objects:
        return commands
    for curr, next in zip(match_objects, match_objects[1:] + [None]):
        if next is None:
            content = action[curr.span()[1] :]
        else:
            content = action[curr.span()[1] : next.span()[0]]
        content = content.lstrip(":")
        command_type = SpecialCommandType(curr.group(0))
        if command_type in [SpecialCommandType.SHELL, SpecialCommandType.PATCH]:
            content = _extract_shell_or_patch(content)
        commands.append(Command(command_type, content))
    return commands
