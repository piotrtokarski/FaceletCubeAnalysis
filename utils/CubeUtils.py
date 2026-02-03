import re


class CubeUtils:
    FACE_ORDER = ['U', 'R', 'F', 'D', 'L', 'B']

    COLOR_TO_FACE = {
        'white': 'D',
        'red': 'L',
        'green': 'F',
        'yellow': 'U',
        'orange': 'R',
        'blue': 'B'
    }

    @staticmethod
    def apply_scramble(cube, scramble):
        cube(scramble)

    @staticmethod
    def clean_solution(solution):
        solution = re.sub(r'\[.*?\]', '', solution)
        return solution.split()

    @staticmethod
    def cube_to_facelet_from_cube(cube):
        facelet = []
        for face in CubeUtils.FACE_ORDER:
            f = cube.get_face(face)
            for row in f:
                for sticker in row:
                    facelet.append(CubeUtils.COLOR_TO_FACE[sticker.colour])
        return ''.join(facelet)

    @staticmethod
    def transform_solution_to_state_list(cube, scramble,solution):
        state_list = []
        CubeUtils.apply_scramble(cube, scramble)
        facelet = CubeUtils.cube_to_facelet_from_cube(cube)
        state_list.append(facelet)

        moves = CubeUtils.clean_solution(solution)
        for i, move in enumerate(moves, start=1):
            cube(move)
            facelet = CubeUtils.cube_to_facelet_from_cube(cube)
            state_list.append(facelet)

        return state_list, moves

    @staticmethod
    def get_redundant_states_markers(state_list):
        if len(state_list) < 2:
            return [0]

        n_moves = len(state_list) - 1
        redundant = [0] * n_moves

        base_states_indexes = list(range(len(state_list)))

        while True:
            start_wrong = None
            stop_wrong = None
            max_length_wrong = 0

            for pos_start, start_index in enumerate(base_states_indexes):
                start_facelet = state_list[start_index]
                for pos_stop in range(pos_start + 1, len(base_states_indexes)):
                    stop_index = base_states_indexes[pos_stop]
                    stop_facelet = state_list[stop_index]

                    if start_facelet != stop_facelet:
                        continue

                    length_wrong = stop_index - start_index
                    if length_wrong > max_length_wrong:
                        max_length_wrong = length_wrong
                        start_wrong = pos_start
                        stop_wrong = pos_stop

            if max_length_wrong == 0:
                break

            for k in range(base_states_indexes[start_wrong], base_states_indexes[stop_wrong]):
                redundant[k] = max(redundant[k], max_length_wrong)

            base_states_indexes = base_states_indexes[:start_wrong] + base_states_indexes[stop_wrong:]

        return [0] + redundant