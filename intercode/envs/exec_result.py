from typing import Union


class ExecResult:
    def __init__(self, content: str, exit_code: int, output: Union[str, bytes] = ""):
        self.content = content
        self.exit_code = exit_code
        if isinstance(output, bytes):
            output = output.decode("utf-8")
        self.output = output
        self.is_valid = exit_code == 0

    def __repr__(self):
        result = (
            f"<ExecResult><CONTENT77>{self.content}<EXIT_CODE77>{self.exit_code}"
            f"<OUTPUT77>{self.output}<IS_VALID77>{self.is_valid}</ExecResult>"
        )
        return result

    @classmethod
    def from_str(cls, str_format: str) -> "ExecResult":
        content = str_format.split("<CONTENT77>")[1].split("<EXIT_CODE77>")[0]
        exit_code = int(str_format.split("<EXIT_CODE77>")[1].split("<OUTPUT77>")[0])
        output = str_format.split("<OUTPUT77>")[1].split("<IS_VALID77>")[0]
        return cls(content, exit_code, output)


class SkipResult(ExecResult):
    def __init__(self, content: str):
        super().__init__(content=content, exit_code=0)
