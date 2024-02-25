"""Run demo for each environment.

Example usage:
python install_all_environments.py
"""

import os

from intercode.envs import SWEEnv
from experiments.logger_helper import Logger


def main():
    logger = Logger(filename=os.path.join("./logs/install/", "install"))
    env = SWEEnv(
        "intercode-swe",
        data_path="./data/swe-bench/ic_swe_bench_dev_sorted.json",
        traj_dir="./traj_dir/",
        logger=logger,
        verbose=True,
    )

    repo_version_pairs = set([])
    for idx in range(len(env.data_loader)):
        record = env.data_loader.get(idx)
        record_info = f"repo: {record['repo']}, version: {record['version']}"
        repo_version_pairs.add(record_info)
    logger.info(f"# of unique_environments: {len(repo_version_pairs)}")
    checked_pairs = set([])
    num_success = 0
    num_failure = 0
    for idx in range(len(env.data_loader)):
        record = env.data_loader.get(idx)
        record_info = f"repo: {record['repo']}, version: {record['version']}"
        if record_info in checked_pairs:
            continue
        checked_pairs.add(record_info)
        logger.info(f"Start {record_info}")
        try:
            env.reset(idx)
            logger.info(f"Successfully installed {record_info}")
            num_success += 1
        except KeyboardInterrupt:
            logger.info("Exiting InterCode environment...")
            env.close()
            break
        except Exception as e:
            logger.error(f"Failed to install {record_info}. Error: {e}")
            num_failure += 1
        finally:
            logger.info(
                f"Finished {len(checked_pairs)}/{len(repo_version_pairs)} repo-version pairs. Success: {num_success}, Failure: {num_failure}"
            )


if __name__ == "__main__":
    main()
