import argparse
import json
import logging
import os
from time import perf_counter
from lm_eval import tasks, evaluator, utils
from huggingface_hub import login
import file_utils as fu



logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", required = False, default = "unknown")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument(
        "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)
    )
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size to try with --batch_size auto",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        dirname = os.path.dirname(args.output_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )
    print(evaluator.make_table(results))

    return args, dumped


if __name__ == "__main__":
    login(token="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")
    start = perf_counter()
    args, results_dict = main()
    end = perf_counter()
    time_count_file = "tests_elapsed_times.json"
    time_count = {}
    model = args.model_args.split("=")[-1]
    elapsed_time = end - start
    task = args.tasks
    machine = args.machine

    fu.add_result_to_csv(results_dict,"runs_info.csv",model,machine,
                            task, elapsed_time)

    if not os.path.exists(time_count_file):
        with open(time_count_file, "w") as json_file:
            json.dump({}, json_file)
    with open(time_count_file, "r") as json_file:
        data = json.load(json_file)
    if model not in data:
        data[model] = {}

    if args.tasks not in data[model]:
        data[model][args.tasks] = []

    def get_run_info(runs: list, args) -> dict:
        found = False        
        for run in runs:
            found = (run["num_fewshot"] == args.num_fewshot and 
                     run["batch_size"] == args.batch_size and
                     run["machine"] == args.machine)
            if found:
                return run
        runs.append({"machine": args.machine,
                    "num_fewshot": args.num_fewshot, 
                    "batch_size": args.batch_size,
                    "elapsed_time": []})
        return runs[-1]

    run_info = get_run_info(data[model][args.tasks], args)
    run_info['elapsed_time'].append(elapsed_time)

    with open(time_count_file, "w") as json_file:
        json.dump(data, json_file, indent=4)

    
