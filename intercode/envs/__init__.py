from intercode.envs.ic_env import (
    IntercodeEnv,
    AGENT_OBS,
    EVAL_OBS,
    CORRUPT_GOLD,
    ACTION_EXEC,
    EXEC_RESULTS,
    AGENT_OBSERVATION,
    REWARD,
)
from intercode.envs.bash.bash_env import BashEnv
from intercode.envs.sql.sql_env import SqlEnv, preprocess_sql
from intercode.envs.ctf.ctf_env import CTFEnv, preprocess_ctf
from intercode.envs.python.python_env import PythonEnv
from intercode.envs.swe.swe_env import SWEEnv


def initialize_env(env_name, **kwargs):
    if env_name == "sql":
        env = SqlEnv(**kwargs, preprocess=preprocess_sql)
    elif env_name == "bash":
        env = BashEnv(**kwargs)
    elif env_name == "python":
        env = PythonEnv(**kwargs, is_agent=True)
    elif env_name == "ctf":
        env = CTFEnv(**kwargs, preprocess=preprocess_ctf)
    elif env_name == "swe":
        env = SWEEnv(**kwargs)
    else:
        raise ValueError(f"Environment {env_name!r} not recognized")
    return env
