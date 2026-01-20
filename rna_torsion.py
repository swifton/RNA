import os
from pymol import cmd

# Set record=True before running: pymol> record=True; run rna_torsion_pymol.py
try:
    record
except NameError:
    record = False

# RNA backbone atoms (in order along the chain)
# Standard naming: P, OP1, OP2, O5', C5', C4', O4', C3', O3', C2', O2', C1'
# For purines (G, A): N9, C4 are used for chi
# For pyrimidines (C, U): N1, C2 are used for chi

def rna_atoms(chain, res, next_res=None, base_type="purine"):
    """Generate atom selection strings for an RNA residue."""
    r = f"{chain}/{res}"
    n = f"{chain}/{next_res}" if next_res else None
    chi_atoms = ("N9", "C4") if base_type == "purine" else ("N1", "C2")
    return {
        "P": f"{r}/P", "OP1": f"{r}/OP1", "OP2": f"{r}/OP2",
        "O5'": f"{r}/O5'", "C5'": f"{r}/C5'", "C4'": f"{r}/C4'",
        "O4'": f"{r}/O4'", "C3'": f"{r}/C3'", "O3'": f"{r}/O3'",
        "C2'": f"{r}/C2'", "O2'": f"{r}/O2'", "C1'": f"{r}/C1'",
        "chi_N": f"{r}/{chi_atoms[0]}", "chi_C": f"{r}/{chi_atoms[1]}",
        "next_P": f"{n}/P" if n else None, "next_OP2": f"{n}/OP2" if n else None,
    }

def rna_torsion_angles(atoms):
    """Define RNA torsion angles from atom dictionaries.

    Alpha, beta, gamma, chi are reversed so 5' end rotates, 3' end stays fixed.
    """
    a = atoms
    angles = {
        "alpha": [a["C5'"], a["O5'"], a["P"], a["OP2"]],
        "beta": [a["C4'"], a["C5'"], a["O5'"], a["P"]],
        "gamma": [a["C3'"], a["C4'"], a["C5'"], a["O5'"]],
        "delta": [a["C5'"], a["C4'"], a["C3'"], a["O3'"]],
        "chi": [a["O4'"], a["C1'"], a["chi_N"], a["chi_C"]],
    }
    if a["next_P"] and a["next_OP2"]:
        angles["epsilon"] = [a["C4'"], a["C3'"], a["O3'"], a["next_P"]]
        angles["zeta"] = [a["C3'"], a["O3'"], a["next_P"], a["next_OP2"]]
    return angles

# Torsion angle definitions for 3G9Y-RNA-short.pdb (3 guanine nucleotides)
atoms_3nt = rna_atoms("C", 2, 3, "purine")
angles_3nt = rna_torsion_angles(atoms_3nt)
torsion_angles_3nt = [angles_3nt["alpha"], angles_3nt["beta"], angles_3nt["gamma"],
                      angles_3nt["epsilon"], angles_3nt["zeta"], angles_3nt["chi"]]

# Torsion angle definitions for 9KTW-RNA-3Cytosine.pdb (3 cytosine nucleotides)
atoms_3cyt = rna_atoms("C", 12, 13, "pyrimidine")
angles_3cyt = rna_torsion_angles(atoms_3cyt)
torsion_angles_3cyt = [angles_3cyt["alpha"], angles_3cyt["beta"], angles_3cyt["gamma"],
                       angles_3cyt["epsilon"], angles_3cyt["zeta"], angles_3cyt["chi"]]

# Torsion angle definitions for 9KTW-protein-3Isoleucine.pdb (3 isoleucine residues, chain B, residues 343-345)
# Protein backbone angles: phi, psi; Side chain angles: chi1, chi2
# Using middle residue (344) for demonstration
# Phi and psi reversed so N-terminus rotates, C-terminus stays fixed
phi_3ile = ["B/344/CA", "B/344/N", "B/343/C", "B/343/O"]
psi_3ile = ["B/344/N", "B/344/CA", "B/344/C", "B/345/N"]
# Chi1 and chi2 reversed so backbone stays fixed, side chain rotates
chi1_3ile = ["B/344/N", "B/344/CA", "B/344/CB", "B/344/CG1"]
chi2_3ile = ["B/344/CA", "B/344/CB", "B/344/CG1", "B/344/CD1"]
torsion_angles_3ile = [phi_3ile, psi_3ile, chi1_3ile, chi2_3ile]

# Select which molecule to use
torsion_angles = angles_3cyt


def dihedral_angle(specs):
    return cmd.get_dihedral(specs[0], specs[1], specs[2], specs[3])


def set_torsion(specs, angle, state=1):
    cmd.set_dihedral(specs[0], specs[1], specs[2], specs[3], angle, state=state)


object_name = cmd.get_names("objects")[0]  # Get the loaded molecule name

frame = 1

def rotate(angle_names=None):
    """Rotate torsion angle(s) through 360 degrees sequentially.

    Args:
        angle_names: List of angle names to rotate (e.g., ["alpha", "beta"]),
                    or None to rotate all angles sequentially.
    """
    global frame
    if angle_names is None:
        angle_names = list(torsion_angles.keys())

    for name in angle_names:
        angle_specs = torsion_angles[name]
        initial_value = int(dihedral_angle(angle_specs))
        for angle in range(initial_value, initial_value + 360, 10):
            cmd.create(object_name, object_name, 1, frame + 1)
            set_torsion(angle_specs, angle, state=frame + 1)
            cmd.unpick()
            frame += 1
        # Reset to initial value for next torsion angle
        cmd.create(object_name, object_name, 1, frame + 1)
        frame += 1


def rotate_simultaneously(angle_names=None, directions=None):
    """Rotate multiple torsion angles simultaneously (crankshaft motion).

    Args:
        angle_names: List of angle names to rotate (e.g., ["alpha", "gamma"]),
                    or None for all angles.
        directions: List of directions (1 or -1) for each angle, or None for all positive.
                   Use opposite directions (e.g., [1, -1]) to visualize compensating rotations.
    """
    global frame
    if angle_names is None:
        angle_names = list(torsion_angles.keys())
    if directions is None:
        directions = [1] * len(angle_names)

    # Get angle specs and initial values for selected angles
    selected_angles = [torsion_angles[name] for name in angle_names]
    selected_initial = [int(dihedral_angle(a)) for a in selected_angles]

    for step in range(0, 360, 1):
        cmd.create(object_name, object_name, 1, frame + 1)
        for angle_specs, initial_value, direction in zip(
            selected_angles, selected_initial, directions
        ):
            new_angle = initial_value + step * direction
            set_torsion(angle_specs, new_angle, state=frame + 1)
            cmd.unpick()
        frame += 1

    # Add final frame reset to initial values
    cmd.create(object_name, object_name, 1, frame + 1)
    frame += 1

rotate_simultaneously(["alpha", "gamma", "zeta"], [1, -1, 1])

# rotate()

# Set up movie from all states and play
cmd.mset(f"1 -{frame}")
cmd.mplay()

if record:
    output_path = os.path.expanduser("~/projects/RNA/movies/test.mp4")
    cmd.movie.produce(output_path, quality=90)
    
