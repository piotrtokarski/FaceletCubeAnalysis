import pycuber as pc
import re
from collections import defaultdict

# =========================
# KONFIGURACJA CSTIMERA
# =========================

FACE_ORDER = ['U', 'R', 'F', 'D', 'L', 'B']

COLOR_TO_FACE = {
    'white':  'D',
    'red':    'L',
    'green':  'F',
    'yellow': 'U',
    'orange': 'R',
    'blue':   'B'
}

# =========================
# FACELET STRING
# =========================

def cube_to_facelet(cube: pc.Cube) -> str:
    facelet = []
    for face in FACE_ORDER:
        f = cube.get_face(face)
        for row in f:
            for sticker in row:
                facelet.append(COLOR_TO_FACE[sticker.colour])
    return ''.join(facelet)


def clean_solution(solution: str) -> list[str]:
    """Usuwa timestampy w [] i rozbija na ruchy"""
    solution = re.sub(r'\[.*?\]', '', solution)
    return solution.split()

# =========================
# LOGIKA ZAKRESÓW
# =========================

def ranges_overlap(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def choose_redundant_ranges(ranges):
    """
    Zasady:
    - jeśli zakresy się pokrywają → wybierz dłuższy
    - jeśli równej długości → wybierz wcześniejszy
    - jeśli się nie pokrywają → zostaw oba
    """
    ranges = sorted(ranges)
    chosen = []

    for current in ranges:
        cur_start, cur_end = current
        cur_len = cur_end - cur_start + 1

        conflict_index = None

        for i, selected in enumerate(chosen):
            if ranges_overlap(current, selected):
                sel_start, sel_end = selected
                sel_len = sel_end - sel_start + 1

                if cur_len > sel_len or (
                    cur_len == sel_len and cur_start < sel_start
                ):
                    conflict_index = i
                else:
                    conflict_index = -1
                break

        if conflict_index is None:
            chosen.append(current)
        elif conflict_index >= 0:
            chosen[conflict_index] = current

    return chosen

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    print("Podaj scramble:")
    scramble = input("> ").strip()

    print("\nPodaj rozwiązanie (z timestampami w []):")
    solution_raw = input("> ").strip()

    try:
        cube = pc.Cube()
        cube(scramble)

        moves = clean_solution(solution_raw)
        facelets_per_move = []

        # =========================
        # SYMULACJA RUCHÓW
        # =========================

        for move in moves:
            cube(move)
            facelets_per_move.append(cube_to_facelet(cube))

        # =========================
        # IDENTYCZNE STANY
        # =========================

        facelet_to_moves = defaultdict(list)

        for move_number, facelet in enumerate(facelets_per_move, start=1):
            facelet_to_moves[facelet].append(move_number)

        redundant_ranges = []

        print("\nFACELETY Z OZNACZENIEM (0 = potrzebny, 1 = zbędny):\n")

        for i, facelet in enumerate(facelets_per_move, start=1):
            flag = 1 if i in redundant_moves else 0
            print(f"{flag} {i}. {facelet}")

        print("\nPOWTARZAJĄCE SIĘ STANY KOSTKI:\n")

        found = False
        for facelet, moves_list in facelet_to_moves.items():
            if len(moves_list) > 1:
                found = True
                moves_str = ", ".join(map(str, moves_list))
                print(f"Ruchy [{moves_str}] mają IDENTYCZNY facelet:")
                print(facelet)
                print()

                first = moves_list[0]
                for later in moves_list[1:]:
                    redundant_ranges.append((first + 1, later))

        if not found:
            print("Brak identycznych faceletów — każdy stan kostki jest unikalny.")

        # =========================
        # WYBÓR OPTYMALNYCH ZAKRESÓW
        # =========================

        if redundant_ranges:
            final_ranges = choose_redundant_ranges(redundant_ranges)

            print("\nZBĘDNE ZAKRESY RUCHÓW (optymalne):\n")
            for start, end in final_ranges:
              length = end - start + 1
              print(f"- Ruchy {start}–{end} są zbędne ({length} ruchów)")


            redundant_moves = set()
            for start, end in final_ranges:
                redundant_moves.update(range(start, end + 1))
        else:
            print("\nBrak zbędnych zakresów ruchów.")
            redundant_moves = set()

    except Exception as e:
        print("\nBłąd:")
        print(e)
