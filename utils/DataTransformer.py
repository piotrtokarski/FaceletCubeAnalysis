import pandas as pd
import pycuber

from utils.CubeUtils import CubeUtils


class DataTransformer:
    @staticmethod
    def prepare(df, verbose=False):
        df_cleaned = df[["id","solution", "scramble"]].dropna()
        total = len(df_cleaned)

        rows = []

        for row_number, (idx, row) in enumerate(df_cleaned.iterrows(), start=1):
            solve_id = str(row["id"])
            scramble = str(row["scramble"]).strip()
            solution = str(row["solution"]).strip()

            if not scramble or not solution:
                if verbose:
                    print(f"[skip] id={solve_id}: empty scramble/solution")
                continue

            cube = pycuber.Cube()
            state_list, moves_list = CubeUtils.transform_solution_to_state_list(cube, scramble,solution)

            redundant_states_full_marker, redundant_states_first_marker = CubeUtils.get_redundant_states_markers(state_list)
            redundant_states_binary_full_marker = [1 if int(redundant_states_full_marker[i]) > 0 else 0 for i in range(len(redundant_states_full_marker))]
            redundant_states_binary_first_marker = [1 if int(redundant_states_first_marker[i]) > 0 else 0 for i in
                                                   range(len(redundant_states_first_marker))]

            rows.append({
                "id": solve_id,
                "scramble": scramble,
                "solution": solution,
                "state_list": state_list,
                "moves_list": moves_list,
                "redundant_states_first": redundant_states_first_marker,
                "redundant_states_binary_first": redundant_states_binary_first_marker,
                "redundant_states": redundant_states_full_marker,
                "redundant_states_binary": redundant_states_binary_full_marker
            })

            # --- progress ---
            if verbose:
                pct = (row_number / total) * 100 if total else 100.0
                print(f"Analyse redundant states progress: {pct:.2f}% ({row_number}/{total})")

        return pd.DataFrame(rows)