#!/usr/bin/env python3
"""
Analyze whether an RNA structure from the rna3db dataset is helical.

A helical RNA structure is characterized by:
1. Regular base stacking (consistent rise and twist between base pairs)
2. Base pairing patterns (Watson-Crick or wobble pairs)
3. Consistent backbone torsion angles typical of A-form helix

This script uses geometric analysis of the backbone to detect helical regions.
"""

import numpy as np
from pathlib import Path
import argparse


def parse_cif_atoms(cif_path):
    """Parse atomic coordinates from a CIF file.

    Returns a dict mapping (residue_num, atom_name) -> (x, y, z)
    """
    atoms = {}
    residue_info = {}  # residue_num -> base_type

    in_atom_section = False
    column_indices = {}

    with open(cif_path, 'r') as f:
        lines = f.readlines()

    # Find column definitions
    for i, line in enumerate(lines):
        if line.startswith('_atom_site.'):
            col_name = line.strip().split('.')[1]
            column_indices[col_name] = len(column_indices)
        elif line.startswith('ATOM') or line.startswith('HETATM'):
            in_atom_section = True
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


def get_backbone_atoms(atoms, residue_nums):
    """Extract backbone P, C4', and C1' atoms for each residue."""
    backbone = {}
    for res_num in residue_nums:
        p = atoms.get((res_num, 'P'))
        c4 = atoms.get((res_num, "C4'"))
        c1 = atoms.get((res_num, "C1'"))

        if p is not None and c4 is not None and c1 is not None:
            backbone[res_num] = {'P': p, "C4'": c4, "C1'": c1}

    return backbone


def calculate_rise_and_twist(backbone, residue_nums):
    """Calculate rise (translation) and twist (rotation) between consecutive residues.

    In an ideal A-form RNA helix:
    - Rise: ~2.8 Å per base pair
    - Twist: ~32.7° per base pair (11 bp per turn)
    """
    rises = []
    p_distances = []

    sorted_residues = sorted(residue_nums)

    for i in range(len(sorted_residues) - 1):
        res1, res2 = sorted_residues[i], sorted_residues[i + 1]

        if res1 not in backbone or res2 not in backbone:
            continue

        # Calculate P-P distance (should be ~5.9 Å for A-form helix)
        p1 = backbone[res1]['P']
        p2 = backbone[res2]['P']
        p_dist = np.linalg.norm(p2 - p1)
        p_distances.append(p_dist)

        # Calculate C1'-C1' distance as a proxy for rise
        c1_1 = backbone[res1]["C1'"]
        c1_2 = backbone[res2]["C1'"]
        rise = np.linalg.norm(c1_2 - c1_1)
        rises.append(rise)

    return rises, p_distances


def calculate_backbone_angles(atoms, residue_nums):
    """Calculate backbone torsion-like angles to assess helical regularity.

    We use pseudo-torsion angles based on P and C4' atoms.
    """
    sorted_residues = sorted(residue_nums)
    angles = []

    for i in range(len(sorted_residues) - 3):
        res_nums = sorted_residues[i:i+4]

        # Get P atoms for 4 consecutive residues
        p_atoms = []
        valid = True
        for res in res_nums:
            if (res, 'P') in atoms:
                p_atoms.append(atoms[(res, 'P')])
            else:
                valid = False
                break

        if not valid or len(p_atoms) != 4:
            continue

        # Calculate pseudo-torsion angle
        angle = dihedral_angle(p_atoms[0], p_atoms[1], p_atoms[2], p_atoms[3])
        angles.append(angle)

    return angles


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
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    m1 = np.cross(n1, b2 / np.linalg.norm(b2))

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return np.degrees(np.arctan2(y, x))


def analyze_helicity(cif_path):
    """Analyze whether the RNA structure is helical.

    Returns a dict with analysis results including:
    - is_helical: Boolean indicating if structure appears helical
    - helical_score: Score from 0-1 indicating helicity
    - metrics: Detailed metrics used for analysis
    """
    atoms, residue_info = parse_cif_atoms(cif_path)

    if not atoms:
        return {
            'is_helical': False,
            'helical_score': 0.0,
            'metrics': {},
            'error': 'No atoms found in file'
        }

    residue_nums = sorted(set(r for r, _ in atoms.keys()))

    if len(residue_nums) < 4:
        return {
            'is_helical': False,
            'helical_score': 0.0,
            'metrics': {'num_residues': len(residue_nums)},
            'error': 'Too few residues for helix analysis'
        }

    backbone = get_backbone_atoms(atoms, residue_nums)
    rises, p_distances = calculate_rise_and_twist(backbone, residue_nums)
    pseudo_torsions = calculate_backbone_angles(atoms, residue_nums)

    # A-form helix criteria:
    # - P-P distance: 5.5-6.5 Å (ideal ~5.9 Å)
    # - C1'-C1' distance: 4.5-6.5 Å
    # - Pseudo-torsion angles should be relatively consistent

    metrics = {
        'num_residues': len(residue_nums),
        'num_backbone_residues': len(backbone),
        'sequence': ''.join(residue_info.get(r, '?') for r in sorted(residue_nums))
    }

    helical_scores = []

    # Score based on P-P distances
    if p_distances:
        mean_p_dist = np.mean(p_distances)
        std_p_dist = np.std(p_distances)
        metrics['mean_p_p_distance'] = mean_p_dist
        metrics['std_p_p_distance'] = std_p_dist

        # Ideal P-P distance is ~5.9 Å for A-form helix
        p_dist_score = max(0, 1 - abs(mean_p_dist - 5.9) / 2.0)
        # Penalize high variance
        p_dist_score *= max(0, 1 - std_p_dist / 2.0)
        helical_scores.append(p_dist_score)

    # Score based on C1'-C1' distances (rise)
    if rises:
        mean_rise = np.mean(rises)
        std_rise = np.std(rises)
        metrics['mean_c1_c1_distance'] = mean_rise
        metrics['std_c1_c1_distance'] = std_rise

        # Ideal C1'-C1' distance is ~5.3 Å for consecutive residues in A-form
        rise_score = max(0, 1 - abs(mean_rise - 5.3) / 2.0)
        rise_score *= max(0, 1 - std_rise / 2.0)
        helical_scores.append(rise_score)

    # Score based on pseudo-torsion angle consistency
    if pseudo_torsions:
        std_torsion = np.std(pseudo_torsions)
        metrics['mean_pseudo_torsion'] = np.mean(pseudo_torsions)
        metrics['std_pseudo_torsion'] = std_torsion

        # Helical structures should have consistent torsion angles
        # Lower std = more regular = more helical
        torsion_score = max(0, 1 - std_torsion / 60.0)
        helical_scores.append(torsion_score)

    # Calculate overall helical score
    if helical_scores:
        helical_score = np.mean(helical_scores)
    else:
        helical_score = 0.0

    # Determine if helical (threshold can be adjusted)
    is_helical = helical_score > 0.5

    return {
        'is_helical': is_helical,
        'helical_score': helical_score,
        'metrics': metrics
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze whether an RNA structure is helical'
    )
    parser.add_argument(
        'cif_file',
        type=str,
        help='Path to the CIF file to analyze'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed metrics'
    )

    args = parser.parse_args()

    cif_path = Path(args.cif_file)
    if not cif_path.exists():
        print(f"Error: File not found: {cif_path}")
        return 1

    print(f"Analyzing: {cif_path.name}")
    print("-" * 50)

    result = analyze_helicity(cif_path)

    if 'error' in result:
        print(f"Error: {result['error']}")
        return 1

    print(f"Helical: {'Yes' if result['is_helical'] else 'No'}")
    print(f"Helical score: {result['helical_score']:.3f}")

    if args.verbose:
        print("\nDetailed metrics:")
        for key, value in result['metrics'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            elif key == 'sequence' and len(value) > 50:
                print(f"  {key}: {value[:50]}... ({len(value)} nt)")
            else:
                print(f"  {key}: {value}")

    return 0


if __name__ == '__main__':
    exit(main())
