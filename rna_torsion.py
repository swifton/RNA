import os
from pymol import cmd

# Set record=True before running: pymol> record=True; run rna_torsion_pymol.py
try:
    record
except NameError:
    record = False

# Torsion angle definitions for 3G9Y-RNA-short.pdb
# Alpha, beta, gamma, chi reversed so 5' end rotates, 3' end stays fixed
chi_3nt = ["C/2/O4'", "C/2/C1'", "C/2/N9", "C/2/C4"]
alpha_3nt = ["C/2/C5'", "C/2/O5'", "C/2/P", "C/2/OP2"]
beta_3nt = ["C/2/C4'", "C/2/C5'", "C/2/O5'", "C/2/P"]
gamma_3nt = ["C/2/C3'", "C/2/C4'", "C/2/C5'", "C/2/O5'"]
delta_3nt = ["C/2/C5'", "C/2/C4'", "C/2/C3'", "C/2/O3'"]
epsilon_3nt = ["C/2/C4'", "C/2/C3'", "C/2/O3'", "C/3/P"]
zeta_3nt = ["C/2/C3'", "C/2/O3'", "C/3/P", "C/3/OP2"]
torsion_angles_3nt = [alpha_3nt, beta_3nt, gamma_3nt, epsilon_3nt, zeta_3nt, chi_3nt]

# Torsion angle definitions for 9KTW-RNA-3Cytosine.pdb (3 cytosine nucleotides, chain C, residues 11-13)
# Chi uses N1 for pyrimidines (C, U)
# Alpha, beta, gamma, chi reversed so 5' end rotates, 3' end stays fixed
chi_3cyt = ["C/12/O4'", "C/12/C1'", "C/12/N1", "C/12/C2"]
alpha_3cyt = ["C/12/C5'", "C/12/O5'", "C/12/P", "C/12/OP2"]
beta_3cyt = ["C/12/C4'", "C/12/C5'", "C/12/O5'", "C/12/P"]
gamma_3cyt = ["C/12/C3'", "C/12/C4'", "C/12/C5'", "C/12/O5'"]
delta_3cyt = ["C/12/C5'", "C/12/C4'", "C/12/C3'", "C/12/O3'"]
epsilon_3cyt = ["C/12/C4'", "C/12/C3'", "C/12/O3'", "C/13/P"]
zeta_3cyt = ["C/12/C3'", "C/12/O3'", "C/13/P", "C/13/OP2"]
torsion_angles_3cyt = [alpha_3cyt, beta_3cyt, gamma_3cyt, epsilon_3cyt, zeta_3cyt, chi_3cyt]

# Torsion angle definitions for 9KTW-protein-3Isoleucine.pdb (3 isoleucine residues, chain B, residues 343-345)
# Protein backbone angles: phi, psi; Side chain angles: chi1, chi2
# Using middle residue (344) for demonstration
# Phi and psi reversed so N-terminus rotates, C-terminus stays fixed
phi_3ile = ["B/344/CA", "B/344/N", "B/343/C", "B/343/O"]
psi_3ile = ["B/345/N", "B/344/C", "B/344/CA", "B/344/N"]
# Chi1 and chi2 reversed so backbone stays fixed, side chain rotates
chi1_3ile = ["B/344/N", "B/344/CA", "B/344/CB", "B/344/CG1"]
chi2_3ile = ["B/344/CA", "B/344/CB", "B/344/CG1", "B/344/CD1"]
torsion_angles_3ile = [phi_3ile, psi_3ile, chi1_3ile, chi2_3ile]

# Select which molecule to use
torsion_angles = torsion_angles_3ile


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

def rotate(angle_index=None):
    """Rotate torsion angle(s) through 360 degrees.

    Args:
        angle_index: Index of the angle to rotate (0-5), or None to rotate all angles sequentially.
    """
    global frame
    if angle_index is not None:
        indices = [angle_index]
    else:
        indices = range(len(torsion_angles))

    for i in indices:
        torsion_angle = torsion_angles[i]
        initial_value = int(dihedral_angle(torsion_angle))
        for angle in range(initial_value, initial_value + 360, 10):
            cmd.create(object_name, object_name, 1, frame + 1)
            set_torsion(torsion_angle, angle, state=frame + 1)
            frame += 1
        # Reset to initial value for next torsion angle
        cmd.create(object_name, object_name, 1, frame + 1)
        frame += 1


rotate()

# Set up movie from all states and play
cmd.mset(f"1 -{frame}")
cmd.mplay()

if record:
    output_path = os.path.expanduser("~/projects/RNA/ChimeraX/movies/test.mp4")
    cmd.movie.produce(output_path, quality=90)
    
