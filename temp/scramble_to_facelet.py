import pycuber as pc

# =========================
# KONFIGURACJA CSTIMERA
# =========================

# Kolejność ścian w facelet stringu csTimera
FACE_ORDER = ['U', 'R', 'F', 'D', 'L', 'B']

# Mapowanie kolorów pycubera -> litery csTimera
COLOR_TO_FACE = {
    'white':  'D',
    'red':    'L',
    'green':  'F',
    'yellow': 'U',
    'orange': 'R',
    'blue':   'B'
}

# =========================
# FACELET STRING (54 znaki)
# =========================

def cube_to_facelet(scramble: str) -> str:
    cube = pc.Cube()
    cube(scramble)

    facelet = []

    for face in FACE_ORDER:
        f = cube.get_face(face)
        for row in f:
            for sticker in row:
                facelet.append(COLOR_TO_FACE[sticker.colour])

    return ''.join(facelet)

if __name__ == "__main__":
    print("Podaj scramble: ")
    scramble = input("> ").strip()

    try:
        facelet = cube_to_facelet(scramble)

        print("\nFACELET STRING:")
        print(facelet)

    except Exception as e:
        print("\n Błąd scrambla:")
        print(e)