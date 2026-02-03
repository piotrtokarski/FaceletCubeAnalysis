import pycuber

from utils.CubeUtils import CubeUtils


class DataTransformer:
    @staticmethod
    def prepare(df):
        df_cleaned = df[["solution", "scramble"]].dropna()


        rows = []

        for idx, row in df_cleaned.iterrows():
            scramble = str(row["scramble"]).strip()
            solution = str(row["solution"]).strip()

            if not scramble or not solution:
                raise ValueError("scramble and solution must not be empty")

            cube = pycuber.Cube()
            state_list = CubeUtils.transform_solution_to_state_list(cube, scramble,solution)

            redundant_moves = CubeUtils.get_redundant_moves_markers(state_list)

            print(redundant_moves)