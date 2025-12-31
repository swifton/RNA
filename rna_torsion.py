import os
from pymol import cmd

# Set record=True before running: pymol> record=True; run rna_torsion_pymol.py
try:
    record
except NameError:
    record = False

# PyMOL atom selections (chain C, residue 2)
chi = ["C/2/C4", "C/2/N9", "C/2/C1'", "C/2/O4'"]
alpha = ["C/2/OP2", "C/2/P", "C/2/O5'", "C/2/C5'"]
beta = ["C/2/P", "C/2/O5'", "C/2/C5'", "C/2/C4'"]
gamma = ["C/2/O5'", "C/2/C5'", "C/2/C4'", "C/2/C3'"]
delta = ["C/2/C5'", "C/2/C4'", "C/2/C3'", "C/2/O3'"]
epsilon = ["C/2/C4'", "C/2/C3'", "C/2/O3'", "C/3/P"]
zeta = ["C/2/C3'", "C/2/O3'", "C/3/P", "C/3/OP2"]
torsion_angles = [alpha, beta, gamma, epsilon, zeta, chi]


def dihedral_angle(specs):
    return cmd.get_dihedral(specs[0], specs[1], specs[2], specs[3])


def set_torsion(specs, angle, state=1):
    cmd.set_dihedral(specs[0], specs[1], specs[2], specs[3], angle, state=state)


# Save initial torsion values
initial_values = []
for atoms in torsion_angles:
    initial_values.append(dihedral_angle(atoms))

object_name = cmd.get_names("objects")[0]  # Get the loaded molecule name

frame = 1

def rotate_individually():
    global frame
    for i in range(len(torsion_angles)):
        torsion_angle = torsion_angles[i]
        initial_value = int(initial_values[i])
        for angle in range(initial_value, initial_value + 360, 10):
            cmd.create(object_name, object_name, 1, frame + 1)
            set_torsion(torsion_angle, angle, state=frame + 1)
            frame += 1
        # Reset to initial value for next torsion angle
        cmd.create(object_name, object_name, 1, frame + 1)
        frame += 1


def rotate_together():
    global frame
    for angle in range(0, 360, 10):
        cmd.create(object_name, object_name, 1, frame + 1)
        for i in range(len(torsion_angles)):
            torsion_angle = torsion_angles[i]
            set_torsion(torsion_angle, angle, state=frame + 1)
        frame += 1


rotate_individually()
rotate_together()

# Hide dihedral angle indicator arrows
cmd.hide("dihedrals")

# Set up movie from all states and play
cmd.mset(f"1 -{frame}")
cmd.mplay()

if record:
    output_path = os.path.expanduser("~/projects/RNA/ChimeraX/movies/test.mp4")
    cmd.movie.produce(output_path, quality=90)

# Restore initial torsion values
for atoms, initial_angle in zip(torsion_angles, initial_values):
    set_torsion(atoms, initial_angle)
