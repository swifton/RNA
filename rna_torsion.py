import sys
from chimerax.core.commands import run
from chimerax.geometry import dihedral

def dihedral_angle(specs):
    # Get your four atoms (example using atom specs)
    atoms = run(session, f"sel {specs[0]} {specs[1]} {specs[2]} {specs[3]}").atoms
    angle = dihedral(atoms[0].coord, atoms[1].coord, atoms[2].coord, atoms[3].coord)
    return angle

def set_torsion(specs, angle):
    run(session, f"torsion {specs[0]} {specs[1]} {specs[2]} {specs[3]} {angle} move small")

record = "--record" in sys.argv

chi = ["/C:2@C4", "/C:2@N9", "/C:2@C1'", "/C:2@O4'"]
alpha = ["/C:2@OP2", "/C:2@P", "/C:2@O5'", "/C:2@C5'"]
beta = ["/C:2@P", "/C:2@O5'", "/C:2@C5'", "/C:2@C4'"]
gamma = ["/C:2@O5'", "/C:2@C5'", "/C:2@C4'", "/C:2@C3'"]
delta = ["/C:2@C5'", "/C:2@C4'", "/C:2@C3'", "/C:2@O3'"]
epsilon = ["/C:2@C4'", "/C:2@C3'", "/C:2@O3'", "/C:3@P"]
zeta = ["/C:2@C3'", "/C:2@O3'", "/C:3@P", "/C:3@OP2"]
torsion_angles = [alpha, beta, gamma, epsilon, zeta, chi]

# Save initial torsion values
initial_values = []
for atoms in torsion_angles:
    initial_values.append(dihedral_angle(atoms))

if record:
    run(session, "movie reset")
    run(session, "movie record")

def rotate_individually():
    for i in range(len(torsion_angles)):
        torsion_angle = torsion_angles[i]
        initial_value = int(initial_values[i])
        for angle in range(initial_value, initial_value + 360, 10):
            set_torsion(torsion_angle, angle)
            run(session, "wait 1")

        set_torsion(torsion_angle, initial_value)


def rotate_together():
    for angle in range(0, 0 + 360, 10):
        for i in range(len(torsion_angles)):
            torsion_angle = torsion_angles[i]
            set_torsion(torsion_angle, angle)
        run(session, "wait 1")

rotate_individually()
rotate_together()

if record:
    run(session, "movie encode ~/projects/RNA/ChimeraX/movies/test.webm framerate 30")

# Restore initial torsion values
for atoms, initial_angle in zip(torsion_angles, initial_values):
    set_torsion(atoms, initial_angle)