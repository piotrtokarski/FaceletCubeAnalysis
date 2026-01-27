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
    """
    Usuwa timestampy w [] i rozbija na ruchy
    """
    solution = re.sub(r'\[.*?\]', '', solution)
    return solution.split()


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

        print("\nFACELETY PO KOLEJNYCH RUCHACH:\n")

        for i, move in enumerate(moves, start=1):
            cube(move)
            facelet = cube_to_facelet(cube)
            facelets_per_move.append(facelet)
            print(f"{i}. {facelet}")

        # =========================
        # WYKRYWANIE IDENTYCZNYCH STANÓW
        # =========================

        facelet_to_moves = defaultdict(list)

        for move_number, facelet in enumerate(facelets_per_move, start=1):
            facelet_to_moves[facelet].append(move_number)

        print("\nPOWTARZAJĄCE SIĘ STANY KOSTKI:\n")

        found = False

        for facelet, moves_list in facelet_to_moves.items():
            if len(moves_list) > 1:
                found = True
                moves_str = ", ".join(map(str, moves_list))
                print(f"Ruchy [{moves_str}] mają IDENTYCZNY facelet:")
                print(facelet)
                print()

        if not found:
            print("Brak identycznych faceletów — każdy stan kostki jest unikalny.")

    except Exception as e:
        print("\nBłąd:")
        print(e)
