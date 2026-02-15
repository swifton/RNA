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

def rna_atoms(chain, res, prev_res=None, next_res=None, base_type="purine"):
    """Generate atom selection strings for an RNA residue."""
    r = f"{chain}/{res}"
    p = f"{chain}/{prev_res}" if prev_res else None
    n = f"{chain}/{next_res}" if next_res else None
    chi_atoms = ("N9", "C4") if base_type == "purine" else ("N1", "C2")
    return {
        "P": f"{r}/P", "OP1": f"{r}/OP1", "OP2": f"{r}/OP2",
        "O5'": f"{r}/O5'", "C5'": f"{r}/C5'", "C4'": f"{r}/C4'",
        "O4'": f"{r}/O4'", "C3'": f"{r}/C3'", "O3'": f"{r}/O3'",
        "C2'": f"{r}/C2'", "O2'": f"{r}/O2'", "C1'": f"{r}/C1'",
        "chi_N": f"{r}/{chi_atoms[0]}", "chi_C": f"{r}/{chi_atoms[1]}",
        "prev_C4'": f"{p}/C4'" if p else None,
        "next_P": f"{n}/P" if n else None, "next_OP2": f"{n}/OP2" if n else None,
        "next_C4'": f"{n}/C4'" if n else None,
    }

def rna_torsion_angles(atoms):
    """Define RNA torsion angles from atom dictionaries.

    All angles are in standard (direct) order: 5' to 3' direction.
    Use flip=-1 in rotate/rotate_simultaneously to flip which side rotates.
    """
    a = atoms
    angles = {
        "alpha": [a["OP2"], a["P"], a["O5'"], a["C5'"]],
        "beta": [a["P"], a["O5'"], a["C5'"], a["C4'"]],
        "gamma": [a["O5'"], a["C5'"], a["C4'"], a["C3'"]],
        "delta": [a["C5'"], a["C4'"], a["C3'"], a["O3'"]],
        "chi": [a["chi_C"], a["chi_N"], a["C1'"], a["O4'"]],
    }
    if a["next_P"] and a["next_OP2"]:
        angles["epsilon"] = [a["C4'"], a["C3'"], a["O3'"], a["next_P"]]
        angles["zeta"] = [a["C3'"], a["O3'"], a["next_P"], a["next_OP2"]]
    # Pseudotorsion angles: eta = C4'(i-1)-P(i)-C4'(i)-P(i+1), theta = P(i)-C4'(i)-P(i+1)-C4'(i+1)
    if a["prev_C4'"] and a["next_P"]:
        angles["eta"] = [a["prev_C4'"], a["P"], a["C4'"], a["next_P"]]
    if a["next_P"] and a["next_C4'"]:
        angles["theta"] = [a["P"], a["C4'"], a["next_P"], a["next_C4'"]]
    return angles


def setup_pseudobonds(atoms_list, object_name):
    """Set up pseudobonds for pseudotorsion angle visualization.

    Creates white bonds connecting C4'-P-C4'-P atoms and removes existing
    backbone bonds so the pseudobonds can rotate freely.

    Args:
        atoms_list: List of atom dictionaries from rna_atoms()
        object_name: Name of the PyMOL object
    """
    # Remove existing backbone bonds that would interfere with pseudotorsion rotation
    for atoms in atoms_list:
        # Remove bonds in the backbone between C4' and P
        # These are: P-O5', O5'-C5', C5'-C4', C3'-O3', O3'-P(next)
        # Keep C4'-C3' bond
        cmd.unbond(atoms["P"], atoms["O5'"])
        cmd.unbond(atoms["O5'"], atoms["C5'"])
        cmd.unbond(atoms["C5'"], atoms["C4'"])
        cmd.unbond(atoms["C3'"], atoms["O3'"])
        if atoms["next_P"]:
            cmd.unbond(atoms["O3'"], atoms["next_P"])

    # Add pseudobonds - use selection that includes both atoms
    for atoms in atoms_list:
        curr_p = atoms["P"]
        curr_c4 = atoms["C4'"]
        next_p = atoms["next_P"]
        if curr_c4:
            selection = f"({curr_c4}) or ({curr_p})"
            cmd.select("_pseudo_tmp", selection)
            cmd.bond("_pseudo_tmp and name C4'", "_pseudo_tmp and name P")
            cmd.delete("_pseudo_tmp")
        # Add pseudobond from current C4' to next P
        if next_p:
            selection = f"({curr_c4}) or ({next_p})"
            cmd.select("_pseudo_tmp", selection)
            cmd.bond("_pseudo_tmp and name C4'", "_pseudo_tmp and name P")
            cmd.delete("_pseudo_tmp")

# Torsion angle definitions for 3G9Y-RNA-short.pdb (3 guanine nucleotides)
atoms_3nt = rna_atoms("C", 2, prev_res=1, next_res=3, base_type="purine")
angles_3nt = rna_torsion_angles(atoms_3nt)
torsion_angles_3nt = [angles_3nt["alpha"], angles_3nt["beta"], angles_3nt["gamma"],
                      angles_3nt["epsilon"], angles_3nt["zeta"], angles_3nt["chi"]]

# Torsion angle definitions for 9KTW-RNA-3Cytosine.pdb (3 cytosine nucleotides)
# Residue 11 (first cytosine) - no prev_res, has next_res=12
atoms_3cyt_11 = rna_atoms("C", 11, prev_res=None, next_res=12, base_type="pyrimidine")
angles_3cyt_11 = rna_torsion_angles(atoms_3cyt_11)

# Residue 12 (middle cytosine) - has prev_res=11, next_res=13
atoms_3cyt_12 = rna_atoms("C", 12, prev_res=11, next_res=13, base_type="pyrimidine")
angles_3cyt_12 = rna_torsion_angles(atoms_3cyt_12)

# Residue 13 (last cytosine) - has prev_res=12, no next_res
atoms_3cyt_13 = rna_atoms("C", 13, prev_res=12, next_res=None, base_type="pyrimidine")
angles_3cyt_13 = rna_torsion_angles(atoms_3cyt_13)

# Combined dictionary with residue numbers in angle names
angles_3cyt = {}
for res, angles in [(11, angles_3cyt_11), (12, angles_3cyt_12), (13, angles_3cyt_13)]:
    for name, specs in angles.items():
        angles_3cyt[f"{name}_{res}"] = specs

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

def rotate(angle_names=None, flips=None, directions=None):
    """Rotate torsion angle(s) through 360 degrees sequentially.

    Args:
        angle_names: List of angle names to rotate (e.g., ["alpha", "beta"]),
                    or None to rotate all angles sequentially.
        flips: List of flips (1 or -1) for each angle, or None for all 1.
               Flip -1 reverses the angle specs so the opposite side of the molecule rotates.
        directions: List of directions (1 or -1) for each angle, or None for all 1.
                   Direction -1 rotates counterclockwise instead of clockwise.
    """
    global frame
    if angle_names is None:
        angle_names = list(torsion_angles.keys())
    if flips is None:
        flips = [1] * len(angle_names)
    if directions is None:
        directions = [1] * len(angle_names)

    for name, flip, direction in zip(angle_names, flips, directions):
        angle_specs = torsion_angles[name]
        if flip == -1:
            angle_specs = angle_specs[::-1]  # Flip which side rotates
        initial_value = int(dihedral_angle(angle_specs))
        for step in range(0, 360 * 5, 5):
            cmd.create(object_name, object_name, 1, frame + 1)
            new_angle = initial_value + step * direction
            set_torsion(angle_specs, new_angle, state=frame + 1)
            cmd.unpick()
            frame += 1
        # Reset to initial value for next torsion angle
        cmd.create(object_name, object_name, 1, frame + 1)
        frame += 1


def rotate_simultaneously(angle_names=None, flips=None, directions=None):
    """Rotate multiple torsion angles simultaneously (crankshaft motion).

    Args:
        angle_names: List of angle names to rotate (e.g., ["alpha", "gamma"]),
                    or None for all angles.
        flips: List of flips (1 or -1) for each angle, or None for all 1.
               Flip -1 reverses the angle specs so the opposite side of the molecule rotates.
        directions: List of directions (1 or -1) for each angle, or None for all 1.
                   Direction -1 rotates counterclockwise instead of clockwise.
                   Use opposite directions (e.g., [1, -1]) to visualize compensating rotations.
    """
    global frame
    if angle_names is None:
        angle_names = list(torsion_angles.keys())
    if flips is None:
        flips = [1] * len(angle_names)
    if directions is None:
        directions = [1] * len(angle_names)

    # Get angle specs (flipped if flip is -1) and initial values
    selected_angles = []
    for name, flip in zip(angle_names, flips):
        angle_specs = torsion_angles[name]
        if flip == -1:
            angle_specs = angle_specs[::-1]  # Flip which side rotates
        selected_angles.append(angle_specs)
    selected_initial = [int(dihedral_angle(a)) for a in selected_angles]

    for step in range(0, 360 * 5, 5):
        cmd.create(object_name, object_name, 1, frame + 1)
        for angle_specs, initial_value, direction in zip(selected_angles, selected_initial, directions):
            new_angle = initial_value + step * direction
            set_torsion(angle_specs, new_angle, state=frame + 1)
            cmd.unpick()
        frame += 1

    # Add final frame reset to initial values
    cmd.create(object_name, object_name, 1, frame + 1)
    frame += 1

# rotate_simultaneously(["alpha_12"], flips=[1], directions=[1])
rotate_simultaneously(["alpha_12", "gamma_12"], flips=[-1, -1], directions=[1, -1])

# setup_pseudobonds([atoms_3cyt_11, atoms_3cyt_12, atoms_3cyt_13], object_name)

# rotate(["theta_12"], [1])

# Set up movie from all states and play
cmd.mset(f"1 -{frame}")
cmd.mplay()

if record:
    output_path = os.path.expanduser("~/projects/RNA/movies/test.mp4")
    cmd.movie.produce(output_path, quality=90)
    
