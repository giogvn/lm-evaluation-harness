import json, os, csv
import pandas as pd
from pathlib import Path


MACHINE = "p3.2xlarge"
STRATEGY = "HF"
LIMIT = 1
RESULTS_CSV = "runs_info.csv"

RESULTS_COLUMNS = ["model", 
                    "task",
                    "version",
                    "machine",
                    "strategy",
                    "task", 
                    "num_fewshot",
                    "limit",
                    "acc",
                    "acc_stderr",
                    "acc_norm",
                    "acc_norm_stderr",
                    "batch_size",
                    "bootstrap_iters",
                    "elapsed_time",
                    "mc1",
                    "mc1_stderr",
                    "mc2",
                    "mc2_stderr"]

TASKS_METRICS = {"truthfulqa_mc": ["mc1",
                                    "mc1_stderr",
                                    "mc2",
                                    "mc2_stderr"],
                 "hendrykcksTest": ["acc",
                                    "acc_stderr",
                                    "acc_norm",
                                    "acc_norm_stderr"]}

RESULTS_LOC = {"results" : 
                    {"task": ["acc",
                              "acc_stderr",
                              "acc_norm",
                              "acc_norm_stderr"
                              "mc1",
                              "mc1_stderr",
                              "mc2",
                              "mc2_stderr"]},

               "versions":{"task": ["version"]},

               "config": ["num_fewshot",
                          "limit",
                          "batch_size",
                          "bootstrap_iters"]}



def fill_args(fillers: list, args: list):
    j = 0
    for i, val in enumerate(args):
        if not val:
            args[i] = fillers[j]
            j +=1
    return args

def mount_run_args(args: dict) -> str:
    out = []
    for key, val in args.items():
        if key == "--model_args='pretrained=":
             val = ','.join(val)
        if type(val) == list:
            val = ' '.join(val)
            args[key] = val
        out.append(key+val)
        
    return ' '.join(out)

def validate_output_path(model: str, task: str, base_path: str = "results") -> str:

    model = model.split('/')
    if len(model) > 1:
        model = model[-1]
    else:
        model = model[0]    
    
    base_path = base_path / Path(model)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    full_path = base_path / task
    with open(full_path, 'w') as f:
        pass
    return str(full_path)

def get_task_few_shot_num(args: dict, task: str) -> str:
    for t, val in args["num_fewshot"].items():
        found = task.find(t)
        if found != -1:
            return val

def already_run(run_args: dict, 
               elapsed_times_path: str,
               n_rounds: int =5) -> bool:

    with open(elapsed_times_path, "r") as f:
        elapsed_times = json.load(f)
    
    model = run_args["--model_args=pretrained="][0]
    task = run_args["--tasks="][0]
    num_fewshot = ["num_fewshot="][0]
    batch_size = ["batch_size="][0]
    machine = ["--machine="][0]

    if model not in elapsed_times:
        return False

    if task not in elapsed_times[model]:
        return False
    if all([arg not in elapsed_times[model][task][0] 
            for arg in [num_fewshot, batch_size, machine]]):
        return len(elapsed_times[model][task][0]["elapsed_time"]) > n_rounds
    
    return True


def run_rounds(run_args: dict, 
               res_path: str = RESULTS_CSV) -> bool:

    if not os.path.exists(res_path):
        return 0
    
    machine = run_args["machine"]
    task = run_args["task"]
    num_fewshot = run_args["num_fewshot"]
    strategy = run_args["strategy"]
    limit = run_args["limit"]
    bootstrap_iters = run_args["bootstrap_iters"]
    df = pd.read_csv(res_path)

    total_runs = df[(df['machine']==machine) & (df['task']==task)
                    & (df['num_fewshot']==num_fewshot) & (df['strategy']==strategy)
                    & (df['limit']==limit) & (df['bootstrap_iters']==bootstrap_iters)]
    
    return total_runs.shape[0]

def add_result_to_csv(results_dict: str,
                       csv_path: str,
                       model: str,
                       machine: str,
                       task: str,
                       elapsed_time: float,
                       strategy: str = STRATEGY,
                       res_struct: dict = RESULTS_LOC,
                       cols: str = RESULTS_COLUMNS) -> None:
    
    new_row = {col: None for col in cols}
    new_row["model"] = model
    new_row["task"] = task
    new_row["machine"] = machine
    new_row["strategy"] = strategy
    new_row["elapsed_time"] = elapsed_time

    results_dict = json.loads(results_dict)

    for arg in res_struct["results"]["task"]:
        if arg not in results_dict["results"][task]:
            new_row[arg] = None
        else: 
            new_row[arg] = results_dict["results"][task][arg]
    for arg in res_struct["versions"]["task"]:
        new_row[arg] = results_dict["versions"][task]
    for arg in res_struct["config"]:
        new_row[arg] = results_dict["config"][arg]    
    
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(cols)

    df = pd.read_csv(csv_path)
    df.loc[len(df)] = new_row
    df.to_csv(csv_path, index=False)

    
            