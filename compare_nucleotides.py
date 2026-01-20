#!/usr/bin/env python3
"""
Compare random nucleotides from the rna3db dataset.

This script:
1. Picks two random nucleotides from two random sequences
2. Calculates differences in torsion angles and pseudotorsion angles
3. Aligns them and calculates RMSD
4. Repeats multiple times and generates scatter plots:
   - Torsion angle difference vs RMSD
   - Pseudotorsion angle difference vs RMSD
"""

import numpy as np
from pathlib import Path
import random
import argparse
import matplotlib.pyplot as plt
from typing import Optional


# RNA backbone torsion angle atom definitions
# alpha: O3'(i-1) - P - O5' - C5'
# beta:  P - O5' - C5' - C4'
# gamma: O5' - C5' - C4' - C3'
# delta: C5' - C4' - C3' - O3'
# epsilon: C4' - C3' - O3' - P(i+1)
# zeta: C3' - O3' - P(i+1) - O5'(i+1)
# chi: O4' - C1' - N9/N1 - C4/C2 (purine/pyrimidine)

TORSION_ATOMS = {
    'alpha': ["O3'_prev", 'P', "O5'", "C5'"],
    'beta': ['P', "O5'", "C5'", "C4'"],
    'gamma': ["O5'", "C5'", "C4'", "C3'"],
    'delta': ["C5'", "C4'", "C3'", "O3'"],
    'epsilon': ["C4'", "C3'", "O3'", 'P_next'],
    'zeta': ["C3'", "O3'", 'P_next', "O5'_next"],
}

# Chi angle atoms depend on base type
CHI_ATOMS_PURINE = ["O4'", "C1'", 'N9', 'C4']  # A, G
CHI_ATOMS_PYRIMIDINE = ["O4'", "C1'", 'N1', 'C2']  # C, U

# Pseudotorsion angles (eta and theta) use P and C4' atoms
# eta: C4'(i-1) - P(i) - C4'(i) - P(i+1)
# theta: P(i) - C4'(i) - P(i+1) - C4'(i+1)
PSEUDOTORSION_ATOMS = {
    'eta': ["C4'_prev", 'P', "C4'", 'P_next'],
    'theta': ['P', "C4'", 'P_next', "C4'_next"],
}

# All backbone atoms needed for a nucleotide (for RMSD calculation)
BACKBONE_ATOMS = ['P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"]


def parse_cif_atoms(cif_path):
    """Parse atomic coordinates from a CIF file.

    Returns a dict mapping (residue_num, atom_name) -> np.array([x, y, z])
    and residue_info mapping residue_num -> base_type
    """
    atoms = {}
    residue_info = {}

    column_indices = {}

    with open(cif_path, 'r') as f:
        lines = f.readlines()

    # Find column definitions
    for line in lines:
        if line.startswith('_atom_site.'):
            col_name = line.strip().split('.')[1]
            column_indices[col_name] = len(column_indices)
        elif line.startswith('ATOM') or line.startswith('HETATM'):
            break

    if not column_indices:
        raise ValueError("Could not find atom_site columns in CIF file")

    # Parse atom records
    for line in lines:
        if not (line.startswith('ATOM') or line.startswith('HETATM')):
            continue

        parts = line.split()
        if len(parts) < max(column_indices.values()) + 1:
            continue

        try:
            atom_name = parts[column_indices.get('label_atom_id', 3)]
            res_num = int(parts[column_indices.get('label_seq_id', 8)])
            comp_id = parts[column_indices.get('label_comp_id', 5)]
            x = float(parts[column_indices.get('Cartn_x', 10)])
            y = float(parts[column_indices.get('Cartn_y', 11)])
            z = float(parts[column_indices.get('Cartn_z', 12)])

            atoms[(res_num, atom_name)] = np.array([x, y, z])
            residue_info[res_num] = comp_id
        except (ValueError, IndexError):
            continue

    return atoms, residue_info


def dihedral_angle(p1, p2, p3, p4):
    """Calculate dihedral angle between 4 points in degrees."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return None

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    m1 = np.cross(n1, b2 / np.linalg.norm(b2))

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return np.degrees(np.arctan2(y, x))


def get_torsion_angles(atoms, residue_info, res_num):
    """Calculate all torsion angles for a nucleotide.

    Returns dict of angle_name -> angle_value (in degrees), or None if atoms missing.
    """
    base_type = residue_info.get(res_num, 'N')
    is_purine = base_type in ('A', 'G')

    angles = {}

    # Get atoms for current, previous, and next residues
    def get_atom(name):
        if name.endswith('_prev'):
            return atoms.get((res_num - 1, name[:-5]))
        elif name.endswith('_next'):
            return atoms.get((res_num + 1, name[:-5]))
        else:
            return atoms.get((res_num, name))

    # Calculate backbone torsion angles
    for angle_name, atom_names in TORSION_ATOMS.items():
        coords = [get_atom(name) for name in atom_names]
        if all(c is not None for c in coords):
            angle = dihedral_angle(*coords)
            if angle is not None:
                angles[angle_name] = angle

    # Calculate chi angle
    chi_atoms = CHI_ATOMS_PURINE if is_purine else CHI_ATOMS_PYRIMIDINE
    coords = [get_atom(name) for name in chi_atoms]
    if all(c is not None for c in coords):
        angle = dihedral_angle(*coords)
        if angle is not None:
            angles['chi'] = angle

    return angles


def get_pseudotorsion_angles(atoms, res_num):
    """Calculate pseudotorsion angles (eta, theta) for a nucleotide.

    Returns dict of angle_name -> angle_value (in degrees).
    """
    angles = {}

    def get_atom(name):
        if name.endswith('_prev'):
            return atoms.get((res_num - 1, name[:-5]))
        elif name.endswith('_next'):
            return atoms.get((res_num + 1, name[:-5]))
        else:
            return atoms.get((res_num, name))

    for angle_name, atom_names in PSEUDOTORSION_ATOMS.items():
        coords = [get_atom(name) for name in atom_names]
        if all(c is not None for c in coords):
            angle = dihedral_angle(*coords)
            if angle is not None:
                angles[angle_name] = angle

    return angles


def get_backbone_coords(atoms, res_num):
    """Get backbone atom coordinates for a nucleotide.

    Returns list of coordinates for RMSD calculation.
    """
    coords = []
    for atom_name in BACKBONE_ATOMS:
        coord = atoms.get((res_num, atom_name))
        if coord is not None:
            coords.append(coord)
    return coords


def kabsch_rmsd(coords1, coords2):
    """Calculate RMSD after optimal superposition using Kabsch algorithm.

    Both inputs should be lists/arrays of 3D coordinates.
    """
    if len(coords1) != len(coords2) or len(coords1) < 3:
        return None

    P = np.array(coords1)
    Q = np.array(coords2)

    # Center both structures
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Compute covariance matrix
    H = P_centered.T @ Q_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, d])
    R = Vt.T @ sign_matrix @ U.T

    # Apply rotation and compute RMSD
    P_rotated = P_centered @ R.T

    diff = P_rotated - Q_centered
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return rmsd


def angle_difference(angle1, angle2):
    """Calculate the difference between two angles, accounting for periodicity.

    Returns the absolute difference in range [0, 180].
    """
    diff = abs(angle1 - angle2)
    if diff > 180:
        diff = 360 - diff
    return diff


def collect_cif_files(dataset_path):
    """Collect all CIF files from the rna3db dataset."""
    cif_files = []
    dataset_path = Path(dataset_path)

    for component_dir in dataset_path.iterdir():
        if not component_dir.is_dir():
            continue
        for structure_dir in component_dir.iterdir():
            if not structure_dir.is_dir():
                continue
            for cif_file in structure_dir.glob('*.cif'):
                cif_files.append(cif_file)

    return cif_files


def get_valid_residues(atoms, residue_info):
    """Get list of residue numbers that have complete backbone atoms."""
    residue_nums = sorted(set(r for r, _ in atoms.keys()))
    valid = []

    for res_num in residue_nums:
        # Check if we have enough atoms for torsion angle calculation
        # Need current residue backbone atoms
        has_backbone = all(
            atoms.get((res_num, atom)) is not None
            for atom in BACKBONE_ATOMS
        )
        if has_backbone:
            valid.append(res_num)

    return valid


def pick_random_nucleotide(cif_files, max_attempts=10):
    """Pick a random nucleotide from a random structure.

    Returns (atoms, residue_info, res_num, cif_path) or None if failed.
    """
    for _ in range(max_attempts):
        cif_path = random.choice(cif_files)
        try:
            atoms, residue_info = parse_cif_atoms(cif_path)
        except Exception:
            continue

        valid_residues = get_valid_residues(atoms, residue_info)
        # Need residues with neighbors for torsion angles
        interior_residues = [r for r in valid_residues
                           if r - 1 in valid_residues and r + 1 in valid_residues]

        if not interior_residues:
            continue

        res_num = random.choice(interior_residues)
        return atoms, residue_info, res_num, cif_path

    return None


def compare_nucleotides(atoms1, residue_info1, res1, atoms2, residue_info2, res2):
    """Compare two nucleotides.

    Returns dict with torsion_diff, pseudotorsion_diff, rmsd, or None if comparison failed.
    """
    # Get torsion angles
    torsions1 = get_torsion_angles(atoms1, residue_info1, res1)
    torsions2 = get_torsion_angles(atoms2, residue_info2, res2)

    # Get pseudotorsion angles
    pseudo1 = get_pseudotorsion_angles(atoms1, res1)
    pseudo2 = get_pseudotorsion_angles(atoms2, res2)

    # Get backbone coordinates for RMSD
    coords1 = get_backbone_coords(atoms1, res1)
    coords2 = get_backbone_coords(atoms2, res2)

    # Calculate RMSD
    rmsd = kabsch_rmsd(coords1, coords2)
    if rmsd is None:
        return None

    # Calculate mean torsion angle difference
    common_torsions = set(torsions1.keys()) & set(torsions2.keys())
    if not common_torsions:
        return None

    torsion_diffs = [angle_difference(torsions1[k], torsions2[k]) for k in common_torsions]
    mean_torsion_diff = np.mean(torsion_diffs)

    # Calculate mean pseudotorsion angle difference
    common_pseudo = set(pseudo1.keys()) & set(pseudo2.keys())
    if not common_pseudo:
        return None

    pseudo_diffs = [angle_difference(pseudo1[k], pseudo2[k]) for k in common_pseudo]
    mean_pseudo_diff = np.mean(pseudo_diffs)

    return {
        'torsion_diff': mean_torsion_diff,
        'pseudotorsion_diff': mean_pseudo_diff,
        'rmsd': rmsd,
        'num_torsions': len(common_torsions),
        'num_pseudotorsions': len(common_pseudo),
    }


def run_comparisons(dataset_path, num_comparisons=100, verbose=False):
    """Run multiple nucleotide comparisons.

    Returns list of comparison results.
    """
    print(f"Collecting CIF files from {dataset_path}...")
    cif_files = collect_cif_files(dataset_path)
    print(f"Found {len(cif_files)} CIF files")

    if len(cif_files) < 2:
        raise ValueError("Need at least 2 CIF files for comparison")

    results = []
    failed = 0

    print(f"Running {num_comparisons} comparisons...")
    for i in range(num_comparisons):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{num_comparisons}")

        # Pick two random nucleotides
        nuc1 = pick_random_nucleotide(cif_files)
        nuc2 = pick_random_nucleotide(cif_files)

        if nuc1 is None or nuc2 is None:
            failed += 1
            continue

        atoms1, residue_info1, res1, path1 = nuc1
        atoms2, residue_info2, res2, path2 = nuc2

        # Compare them
        result = compare_nucleotides(atoms1, residue_info1, res1,
                                    atoms2, residue_info2, res2)

        if result is not None:
            result['path1'] = str(path1.name)
            result['path2'] = str(path2.name)
            result['res1'] = res1
            result['res2'] = res2
            results.append(result)
        else:
            failed += 1

    print(f"Completed {len(results)} successful comparisons ({failed} failed)")
    return results


def create_plots(results, output_prefix='nucleotide_comparison'):
    """Create scatter plots of angle differences vs RMSD."""
    if not results:
        print("No results to plot")
        return

    torsion_diffs = [r['torsion_diff'] for r in results]
    pseudo_diffs = [r['pseudotorsion_diff'] for r in results]
    rmsds = [r['rmsd'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Torsion angle difference vs RMSD
    ax1 = axes[0]
    ax1.scatter(torsion_diffs, rmsds, alpha=0.5, edgecolors='none', s=30)
    ax1.set_xlabel('Mean Torsion Angle Difference (degrees)', fontsize=12)
    ax1.set_ylabel('RMSD (Angstroms)', fontsize=12)
    ax1.set_title('Torsion Angle Difference vs RMSD', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Add correlation coefficient
    corr1 = np.corrcoef(torsion_diffs, rmsds)[0, 1]
    ax1.text(0.05, 0.95, f'r = {corr1:.3f}', transform=ax1.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Pseudotorsion angle difference vs RMSD
    ax2 = axes[1]
    ax2.scatter(pseudo_diffs, rmsds, alpha=0.5, edgecolors='none', s=30, color='orange')
    ax2.set_xlabel('Mean Pseudotorsion Angle Difference (degrees)', fontsize=12)
    ax2.set_ylabel('RMSD (Angstroms)', fontsize=12)
    ax2.set_title('Pseudotorsion Angle Difference vs RMSD', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Add correlation coefficient
    corr2 = np.corrcoef(pseudo_diffs, rmsds)[0, 1]
    ax2.text(0.05, 0.95, f'r = {corr2:.3f}', transform=ax2.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    output_path = f'{output_prefix}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Compare random nucleotides from rna3db dataset'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='datasets/rna3db/train_set',
        help='Path to rna3db train_set directory'
    )
    parser.add_argument(
        '--num-comparisons', '-n',
        type=int,
        default=500,
        help='Number of nucleotide comparisons to perform'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='nucleotide_comparison',
        help='Output prefix for plot files'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print progress during comparisons'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        return 1

    # Run comparisons
    results = run_comparisons(dataset_path, args.num_comparisons, args.verbose)

    if not results:
        print("Error: No successful comparisons")
        return 1

    # Print summary statistics
    torsion_diffs = [r['torsion_diff'] for r in results]
    pseudo_diffs = [r['pseudotorsion_diff'] for r in results]
    rmsds = [r['rmsd'] for r in results]

    print("\nSummary Statistics:")
    print(f"  Torsion angle diff: mean={np.mean(torsion_diffs):.1f}, "
          f"std={np.std(torsion_diffs):.1f}, range=[{np.min(torsion_diffs):.1f}, {np.max(torsion_diffs):.1f}]")
    print(f"  Pseudotorsion diff: mean={np.mean(pseudo_diffs):.1f}, "
          f"std={np.std(pseudo_diffs):.1f}, range=[{np.min(pseudo_diffs):.1f}, {np.max(pseudo_diffs):.1f}]")
    print(f"  RMSD: mean={np.mean(rmsds):.2f}, "
          f"std={np.std(rmsds):.2f}, range=[{np.min(rmsds):.2f}, {np.max(rmsds):.2f}]")

    # Create plots
    create_plots(results, args.output)

    return 0


if __name__ == '__main__':
    exit(main())
