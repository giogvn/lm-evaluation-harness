import pandas as pd


class CostsCalculator:
    def __init__(self, data_dict: dict, columns: list[str]) -> None:
        self.data_dict = data_dict
        self.data = {col: [] for col in columns}
        self.df = None

    def add_costs_column(
        self,
        costs_df: pd.DataFrame,
        cost_col: str = "cost(US$/hour)",
        runs_df: pd.DataFrame = None,
        convert_s_to_h: bool = True,
    ) -> pd.DataFrame:
        if runs_df is None:
            runs_df = self.df
        if convert_s_to_h:
            convert = lambda x: x / 3600
        else:
            convert = lambda x: x
        if "cost" not in self.data:
            self.data["cost(US$)"] = []

        for _, machine_config in costs_df.iterrows():
            for run_index, run in runs_df.iterrows():
                if machine_config["machine"] == run["machine"]:
                    cost = machine_config[cost_col] * convert(
                        run["elapsed_time"]
                    )
                    runs_df.loc[run_index, "cost(US$)"] = cost

        return runs_df

    def _assemble_values(self) -> list:
        """Gets the values of a dictionary.

        Args:
            dictionary (dict): Dictionary.
            key (str): Key.

        Returns:
            list: List of values.
        """
        for model, task_keys in self.data_dict.items():
            for task, values in task_keys.items():
                for run in values:
                    for t in run["elapsed_time"]:
                        self.data["model"].append(model)
                        self.data["task"].append(task)
                        self.data["machine"].append(run["machine"])
                        self.data["num_fewshot"].append(run["num_fewshot"])
                        self.data["batch_size"].append(run["batch_size"])
                        self.data["elapsed_time"].append(t)

    def _create_df(self) -> pd.DataFrame:
        out = pd.DataFrame(self.data)
        self.df = out
        return out

    def get_elapsed_times_df(self) -> pd.DataFrame:
        """Converts a dictionary of elapsed times to a pandas DataFrame.

        Args:
            elapsed_times (dict): Dictionary of elapsed times.
            columns (list[str]): List of column names.

        Returns:
            pd.DataFrame: Pandas DataFrame.
        """
        self._assemble_values()
        return self._create_df()
