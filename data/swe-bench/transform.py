import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_swe_bench", type=str, help="Path to SWE-bench dataset (.json file)"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        help="Output agent bench dataset name",
        default="ic_swe_bench",
    )
    args = parser.parse_args()

    dataset_swe_bench = []
    with open(args.path_swe_bench, "r") as f:
        tasks = json.load(f)
        for t in tasks:
            dataset_swe_bench.append(
                {
                    "query": t["problem_statement"],
                    "task_id": t["instance_id"],
                    "base_commit": t["base_commit"],
                    "repo": t["repo"],
                    "patch": t["patch"],
                    "tests": {
                        "patch": t["test_patch"],
                        "FAIL_TO_PASS": t["FAIL_TO_PASS"],
                        "PASS_TO_PASS": t["PASS_TO_PASS"],
                    },
                    "version": t["version"],
                    "environment_setup_commit": t["environment_setup_commit"],
                }
            )

    with open(f"{args.output_name}.json", "w") as f:
        json.dump(dataset_swe_bench, fp=f)
