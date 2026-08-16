# enhanced_rna_processing.py


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Bio import PDB
from Bio.PDB.vectors import calc_dihedral, calc_angle
import pickle
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import warnings
import networkx as nx


from utils import positional_encoding


warnings.filterwarnings("ignore", category=PDB.PDBExceptions.PDBConstructionWarning)


rna_dict = {
    'A': [1, 0, 0, 0, 0],
    'U': [0, 1, 0, 0, 0],
    'G': [0, 0, 1, 0, 0],
    'C': [0, 0, 0, 1, 0],
    'N': [0, 0, 0, 0, 1],
    'ADE': [1, 0, 0, 0, 0],
    'URI': [0, 1, 0, 0, 0],
    'GUA': [0, 0, 1, 0, 0],
    'CYT': [0, 0, 0, 1, 0],
    'I': [0, 0, 0, 0, 1],
    'PSU': [0, 0, 0, 0, 1],
    'T': [0, 1, 0, 0, 0],
}


ss_type_encoding = {
    'stem': [1, 0, 0, 0, 0],
    'hairpin': [0, 1, 0, 0, 0],
    'internal_loop': [0, 0, 1, 0, 0],
    'bulge': [0, 0, 0, 1, 0],
    'multi_branch': [0, 0, 0, 0, 1],
    'unknown': [0, 0, 0, 0, 0]
}


purine_types = ['A', 'G', 'ADE', 'GUA']
pyrimidine_types = ['U', 'C', 'URI', 'CYT', 'T']


def get_residue_atoms(residue):
    'Get residue atoms.'
    atom_coords = {}
    atom_list = ["P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]


    resname = residue.get_resname()
    if resname in ['A', 'G', 'ADE', 'GUA']:
        atom_list.extend(["N9", "C8", "N7", "C5", "C6", "N1", "C2"])
        if resname in ['A', 'ADE']:
            atom_list.extend(["N6", "C4", "N3"])
        elif resname in ['G', 'GUA']:
            atom_list.extend(["O6", "C4", "N3", "N2"])
    elif resname in ['U', 'C', 'URI', 'CYT', 'T']:
        atom_list.extend(["N1", "C2", "N3", "C4", "C5", "C6"])
        if resname in ['U', 'URI', 'T']:
            atom_list.extend(["O4", "O2"])
        elif resname in ['C', 'CYT']:
            atom_list.extend(["N4", "O2"])

    for atom_name in atom_list:
        if atom_name in residue:
            atom_coords[atom_name] = residue[atom_name].coord

    return atom_coords


def calculate_rna_torsion_angles(residue, prev_residue=None, next_residue=None):
    'Calculate rna torsion angles.'
    torsion_angles = {
        'alpha': np.nan,
        'beta': np.nan,
        'gamma': np.nan,
        'delta': np.nan,
        'epsilon': np.nan,
        'zeta': np.nan,
        'chi': np.nan,
        'pseudorotation': np.nan
    }


    atoms = get_residue_atoms(residue)


    if prev_residue and "O3'" in atoms:
        prev_atoms = get_residue_atoms(prev_residue)

        # α: O3'(i-1)-P-O5'-C5'
        if all(atom in atoms for atom in ["P", "O5'", "C5'"]) and "O3'" in prev_atoms:
            try:
                torsion_angles['alpha'] = calc_dihedral(
                    PDB.Vector(prev_atoms["O3'"]),
                    PDB.Vector(atoms["P"]),
                    PDB.Vector(atoms["O5'"]),
                    PDB.Vector(atoms["C5'"])
                )
            except:
                pass

        # β: P-O5'-C5'-C4'
        if all(atom in atoms for atom in ["P", "O5'", "C5'", "C4'"]):
            try:
                torsion_angles['beta'] = calc_dihedral(
                    PDB.Vector(atoms["P"]),
                    PDB.Vector(atoms["O5'"]),
                    PDB.Vector(atoms["C5'"]),
                    PDB.Vector(atoms["C4'"])
                )
            except:
                pass

        # γ: O5'-C5'-C4'-C3'
        if all(atom in atoms for atom in ["O5'", "C5'", "C4'", "C3'"]):
            try:
                torsion_angles['gamma'] = calc_dihedral(
                    PDB.Vector(atoms["O5'"]),
                    PDB.Vector(atoms["C5'"]),
                    PDB.Vector(atoms["C4'"]),
                    PDB.Vector(atoms["C3'"])
                )
            except:
                pass

    # δ: C5'-C4'-C3'-O3'
    if all(atom in atoms for atom in ["C5'", "C4'", "C3'", "O3'"]):
        try:
            torsion_angles['delta'] = calc_dihedral(
                PDB.Vector(atoms["C5'"]),
                PDB.Vector(atoms["C4'"]),
                PDB.Vector(atoms["C3'"]),
                PDB.Vector(atoms["O3'"])
            )
        except:
            pass


    if next_residue and "O3'" in atoms:
        next_atoms = get_residue_atoms(next_residue)

        # ε: C4'-C3'-O3'-P(i+1)
        if all(atom in atoms for atom in ["C4'", "C3'", "O3'"]) and "P" in next_atoms:
            try:
                torsion_angles['epsilon'] = calc_dihedral(
                    PDB.Vector(atoms["C4'"]),
                    PDB.Vector(atoms["C3'"]),
                    PDB.Vector(atoms["O3'"]),
                    PDB.Vector(next_atoms["P"])
                )
            except:
                pass

        # ζ: C3'-O3'-P(i+1)-O5'(i+1)
        if all(atom in atoms for atom in ["C3'", "O3'"]) and all(atom in next_atoms for atom in ["P", "O5'"]):
            try:
                torsion_angles['zeta'] = calc_dihedral(
                    PDB.Vector(atoms["C3'"]),
                    PDB.Vector(atoms["O3'"]),
                    PDB.Vector(next_atoms["P"]),
                    PDB.Vector(next_atoms["O5'"])
                )
            except:
                pass


    resname = residue.get_resname()
    base_atom = "N9" if resname in purine_types else "N1"

    if all(atom in atoms for atom in ["O4'", "C1'", base_atom]):
        try:
            torsion_angles['chi'] = calc_dihedral(
                PDB.Vector(atoms["O4'"]),
                PDB.Vector(atoms["C1'"]),
                PDB.Vector(atoms[base_atom]),
                PDB.Vector(atoms["C2" if base_atom == "N1" else "C4"])
            )
        except:
            pass


    if all(atom in atoms for atom in ["C1'", "C2'", "C3'", "C4'", "O4'"]):
        try:

            v1 = calc_dihedral(
                PDB.Vector(atoms["C4'"]),
                PDB.Vector(atoms["O4'"]),
                PDB.Vector(atoms["C1'"]),
                PDB.Vector(atoms["C2'"])
            )
            v2 = calc_dihedral(
                PDB.Vector(atoms["O4'"]),
                PDB.Vector(atoms["C1'"]),
                PDB.Vector(atoms["C2'"]),
                PDB.Vector(atoms["C3'"])
            )
            v3 = calc_dihedral(
                PDB.Vector(atoms["C1'"]),
                PDB.Vector(atoms["C2'"]),
                PDB.Vector(atoms["C3'"]),
                PDB.Vector(atoms["C4'"])
            )
            v4 = calc_dihedral(
                PDB.Vector(atoms["C2'"]),
                PDB.Vector(atoms["C3'"]),
                PDB.Vector(atoms["C4'"]),
                PDB.Vector(atoms["O4'"])
            )
            v5 = calc_dihedral(
                PDB.Vector(atoms["C3'"]),
                PDB.Vector(atoms["C4'"]),
                PDB.Vector(atoms["O4'"]),
                PDB.Vector(atoms["C1'"])
            )


            # P = arctan((v2+v5)-(v3+v4), 2*v1*(sin(36°)+sin(72°)))

            numerator = (v2 + v5) - (v3 + v4)
            denominator = 2 * v1 * (np.sin(np.radians(36)) + np.sin(np.radians(72)))
            p_angle = np.arctan2(numerator, denominator)
            torsion_angles['pseudorotation'] = np.degrees(p_angle) % 360

        except:
            pass

    return torsion_angles


def detect_ribose_puckering(residue):
    'Detect ribose puckering.'

    default_result = [0, 0]


    required_atoms = ["C1'", "C2'", "C3'", "C4'", "O4'"]
    atoms = {}

    for atom_name in required_atoms:
        if atom_name in residue:
            atoms[atom_name] = residue[atom_name].get_vector()
        else:
            return default_result

    try:

        torsion_angles = calculate_rna_torsion_angles(residue)
        p_angle = torsion_angles['pseudorotation']

        if not np.isnan(p_angle):


            if 0 <= p_angle < 90 or 270 <= p_angle < 360:
                return [1, 0]  # C3'-endo
            elif 90 <= p_angle < 270:
                return [0, 1]  # C2'-endo



        plane_normal = (atoms["O4'"] - atoms["C4'"]).cross(atoms["C1'"] - atoms["O4'"]).normalized()


        c2_to_plane = (atoms["C2'"] - atoms["C4'"]).dot(plane_normal)

        if c2_to_plane > 0:
            return [0, 1]  # C2'-endo
        else:
            return [1, 0]  # C3'-endo

    except:

        try:

            c3_z = residue["C3'"].get_coord()[2]
            c2_z = residue["C2'"].get_coord()[2]

            if c3_z > c2_z:
                return [1, 0]  # C3'-endo
            else:
                return [0, 1]  # C2'-endo
        except:
            pass

    return default_result


def predict_secondary_structure_type(residue_idx, chain_length, base_pairs):
    'Predict secondary structure type.'

    default_result = ss_type_encoding['unknown']


    paired_with = None
    try:
        for i, j in base_pairs:
            if i == residue_idx:
                paired_with = j
                break
            elif j == residue_idx:
                paired_with = i
                break
    except ValueError as e:
        print(f"Warning: could not unpack a base pair during secondary-structure prediction: {e}")
        return default_result


    if paired_with is None:

        in_loop = False
        for i, j in base_pairs:

            if i < residue_idx < j or j < residue_idx < i:
                in_loop = True
                break

        if in_loop:


            return ss_type_encoding['hairpin']
        else:
            return default_result



    adjacent_paired = False
    if residue_idx + 1 < chain_length and paired_with - 1 >= 0:
        for i, j in base_pairs:
            if (i == residue_idx + 1 and j == paired_with - 1) or (j == residue_idx + 1 and i == paired_with - 1):
                adjacent_paired = True
                break

    if adjacent_paired:
        return ss_type_encoding['stem']



    return ss_type_encoding['stem']


def detect_base_pairs(structure, distance_threshold=4.0, angle_threshold=35.0):
    'Detect base pairs.'
    base_pairs = []
    base_pairs_with_type = []


    nucleotides = []
    for model in structure:
        for chain in model:
            for residue in chain:

                if residue.get_resname() in list(rna_dict.keys()):
                    nucleotides.append(residue)


    for i, nt1 in enumerate(nucleotides):
        for j, nt2 in enumerate(nucleotides[i + 1:], i + 1):

            if abs(nt1.id[1] - nt2.id[1]) <= 1 and nt1.parent.id == nt2.parent.id:
                continue


            pair_type = check_base_pairing(nt1, nt2, distance_threshold)
            if pair_type:
                base_pairs.append((i, j))
                base_pairs_with_type.append((i, j, pair_type))

    return base_pairs, base_pairs_with_type


def check_base_pairing(nt1, nt2, distance_threshold=4.0):
    'Check base pairing.'

    name1 = nt1.get_resname()
    name2 = nt2.get_resname()


    if name1 in ['A', 'ADE'] and name2 in ['U', 'URI', 'T']:

        if 'N1' in nt1 and 'N3' in nt2:
            dist = np.linalg.norm(nt1['N1'].coord - nt2['N3'].coord)
            if dist <= distance_threshold:
                return 'WC'

    elif name1 in ['U', 'URI', 'T'] and name2 in ['A', 'ADE']:

        if 'N3' in nt1 and 'N1' in nt2:
            dist = np.linalg.norm(nt1['N3'].coord - nt2['N1'].coord)
            if dist <= distance_threshold:
                return 'WC'

    elif name1 in ['G', 'GUA'] and name2 in ['C', 'CYT']:

        if 'N1' in nt1 and 'N3' in nt2:
            dist = np.linalg.norm(nt1['N1'].coord - nt2['N3'].coord)
            if dist <= distance_threshold:
                return 'WC'

    elif name1 in ['C', 'CYT'] and name2 in ['G', 'GUA']:

        if 'N3' in nt1 and 'N1' in nt2:
            dist = np.linalg.norm(nt1['N3'].coord - nt2['N1'].coord)
            if dist <= distance_threshold:
                return 'WC'

    elif name1 in ['G', 'GUA'] and name2 in ['U', 'URI', 'T']:

        if 'N1' in nt1 and 'O2' in nt2:
            dist = np.linalg.norm(nt1['N1'].coord - nt2['O2'].coord)
            if dist <= distance_threshold:
                return 'wobble'

    elif name1 in ['U', 'URI', 'T'] and name2 in ['G', 'GUA']:

        if 'O2' in nt1 and 'N1' in nt2:
            dist = np.linalg.norm(nt1['O2'].coord - nt2['N1'].coord)
            if dist <= distance_threshold:
                return 'wobble'



    return None


def calculate_solvent_accessibility(residue, probe_radius=1.4):
    'Calculate solvent accessibility.'




    atoms = [atom for atom in residue]
    if not atoms:
        return 0.5


    center = np.mean([atom.coord for atom in atoms], axis=0)


    avg_distance = np.mean([np.linalg.norm(atom.coord - center) for atom in atoms])



    accessibility = min(1.0, avg_distance / 10.0)

    return accessibility


def extract_rna_features(residue, prev_residue=None, next_residue=None, base_pairs=None, residue_idx=None,
                         chain_length=None):
    'Extract rna features.'
    features = []


    resname = residue.get_resname()
    nucleotide_type = rna_dict.get(resname, rna_dict['N'])
    features.extend(nucleotide_type)


    puckering = detect_ribose_puckering(residue)
    features.extend(puckering)


    if base_pairs is not None and residue_idx is not None and chain_length is not None:
        try:
            ss_feature = predict_secondary_structure_type(residue_idx, chain_length, base_pairs)
        except Exception as e:
            print(f"Warning: could not predict the secondary-structure type: {e}")
            ss_feature = ss_type_encoding['unknown']
    else:
        ss_feature = ss_type_encoding['unknown']
    features.extend(ss_feature)


    accessibility = calculate_solvent_accessibility(residue)
    features.append(accessibility)


    torsion_angles = calculate_rna_torsion_angles(residue, prev_residue, next_residue)


    normalized_angles = []
    for angle_name in ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi']:
        angle = torsion_angles.get(angle_name, np.nan)
        if not np.isnan(angle):

            norm_angle = np.cos(np.radians(angle))
            normalized_angles.append(norm_angle)

            norm_angle_sin = np.sin(np.radians(angle))
            normalized_angles.append(norm_angle_sin)
        else:
            normalized_angles.extend([0, 0])

    features.extend(normalized_angles)


    pseudo_angle = torsion_angles.get('pseudorotation', np.nan)
    if not np.isnan(pseudo_angle):
        features.append(np.cos(np.radians(pseudo_angle)))
        features.append(np.sin(np.radians(pseudo_angle)))
    else:
        features.extend([0, 0])


    if base_pairs is not None and residue_idx is not None:
        is_paired = 0
        try:
            for i, j in base_pairs:
                if i == residue_idx or j == residue_idx:
                    is_paired = 1
                    break
        except ValueError as e:
            print(f"Warning: base-pair validation failed: {e}")
            is_paired = 0
        features.append(is_paired)
    else:
        features.append(0)



    is_purine = 1 if resname in purine_types else 0
    is_pyrimidine = 1 if resname in pyrimidine_types else 0
    features.extend([is_purine, is_pyrimidine])



    features.append(1 if "P" in residue else 0)

    return np.array(features)


def create_rna_graph_from_structure(structure, threshold=8.0, include_base_pairs=True):
    'Create rna graph from structure.'
    import networkx as nx


    G = nx.Graph()


    residues = []
    residue_coords = {}

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in list(rna_dict.keys()):
                    res_id = (chain.id, residue.id)
                    residues.append(residue)


                    atoms = [atom.coord for atom in residue]
                    if atoms:
                        center_coord = np.mean(atoms, axis=0)
                        residue_coords[res_id] = center_coord


                    G.add_node(res_id, residue=residue)


    if include_base_pairs:
        try:
            base_pairs, base_pairs_with_type = detect_base_pairs(structure)
        except Exception as e:
            print(f"Warning: base-pair detection failed: {e}")
            base_pairs = []
            base_pairs_with_type = []
    else:
        base_pairs = []
        base_pairs_with_type = []


    chain_length = len(residues)
    for i, residue in enumerate(residues):
        res_id = (residue.get_parent().id, residue.id)


        prev_residue = residues[i - 1] if i > 0 else None
        next_residue = residues[i + 1] if i < len(residues) - 1 else None


        features = extract_rna_features(
            residue,
            prev_residue,
            next_residue,
            base_pairs=base_pairs,
            residue_idx=i,
            chain_length=chain_length
        )


        G.nodes[res_id]['feature'] = features


    for i, res_i in enumerate(residues):
        id_i = (res_i.get_parent().id, res_i.id)
        coord_i = residue_coords.get(id_i)

        if coord_i is None:
            continue

        for j, res_j in enumerate(residues[i + 1:], i + 1):
            id_j = (res_j.get_parent().id, res_j.id)
            coord_j = residue_coords.get(id_j)

            if coord_j is None:
                continue


            distance = np.linalg.norm(coord_i - coord_j)


            if distance <= threshold:

                edge_feature = positional_encoding(distance)
                G.add_edge(id_i, id_j, distance=distance, feature=edge_feature)


    for i, j, pair_type in base_pairs_with_type:
        id_i = (residues[i].get_parent().id, residues[i].id)
        id_j = (residues[j].get_parent().id, residues[j].id)


        if G.has_edge(id_i, id_j):

            G[id_i][id_j]['base_pair'] = pair_type
        else:


            distance = np.linalg.norm(residue_coords[id_i] - residue_coords[id_j])
            edge_feature = positional_encoding(distance)
            G.add_edge(id_i, id_j, distance=distance, feature=edge_feature, base_pair=pair_type)

    return G


def visualize_contact_map(contact_map, residue_info, output_file=None):
    'Visualize contact map.'
    plt.figure(figsize=(10, 8))
    plt.imshow(contact_map, cmap='viridis')
    plt.colorbar(label='Contact')


    n_residues = len(residue_info)
    step = max(1, n_residues // 20)
    plt.xticks(np.arange(0, n_residues, step),
               [f"{info[0]}:{info[1][1]}" for info in residue_info[::step]],
               rotation=90)
    plt.yticks(np.arange(0, n_residues, step),
               [f"{info[0]}:{info[1][1]}" for info in residue_info[::step]])

    plt.title(f'RNA Contact Map (threshold: 8Å)')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Contact map saved to {output_file}")
    else:
        plt.show()

    plt.close()

def visualize_rna_graph(G, output_file=None):
    'Visualize rna graph.'
    import matplotlib.pyplot as plt
    import networkx as nx


    plt.figure(figsize=(12, 10))


    pos = nx.spring_layout(G, seed=42)


    nx.draw_networkx_nodes(G, pos, node_size=50)



    normal_edges = [(u, v) for u, v in G.edges() if 'base_pair' not in G[u][v]]
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, width=1, alpha=0.5)


    base_pair_edges = [(u, v) for u, v in G.edges() if 'base_pair' in G[u][v]]
    if base_pair_edges:
        nx.draw_networkx_edges(G, pos, edgelist=base_pair_edges, width=2,
                               edge_color='red', style='dashed')


    labels = {node: f"{node[0]}:{node[1][1]}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.title("RNA Structure Graph Representation")
    plt.axis('off')

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Graph visualization saved to {output_file}")
    else:
        plt.show()

    plt.close()

def process_rna_pdbs(directory, output_dir='./rna_analysis'):
    'Process rna pdbs.'
    parser = PDB.PDBParser(QUIET=True)
    results = {}


    os.makedirs(output_dir, exist_ok=True)


    rna_pdbs = list(Path(directory).glob('*_nucleic.pdb'))


    with tqdm(total=len(rna_pdbs),
              desc="Processing RNA structures",
              unit="file",
              ncols=100,
              colour='blue',
              file=sys.stdout) as pbar:

        for pdb_path in rna_pdbs:
            pdb_id = pdb_path.stem.split('_')[0]


            file_size_mb = pdb_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 1:
                print(f"Skipping large file {pdb_id} ({file_size_mb:.2f} MB > 1 MB)")
                pbar.update(1)
                continue
            try:

                structure = parser.get_structure(pdb_id, pdb_path)


                contact_map, residue_info = rna_contact_map(structure, threshold=8.0)


                try:
                    graph = create_rna_graph_from_structure(structure, threshold=8.0)
                except Exception as e:
                    print(f"Warning: could not create a graph for {pdb_id}: {e}")

                    graph = nx.Graph()
                    for idx, info in enumerate(residue_info):
                        graph.add_node(info, feature=np.zeros(30))


                results[pdb_id] = {
                    'contact_map': contact_map,
                    'residue_info': residue_info,
                    'graph': graph,
                    'pdb_path': str(pdb_path)
                }


                visualize_contact_map(
                    contact_map,
                    residue_info,
                    output_file=os.path.join(output_dir, f"{pdb_id}_contact_map.png")
                )


                visualize_rna_graph(
                    graph,
                    output_file=os.path.join(output_dir, f"{pdb_id}_graph.png")
                )


                node_features = {node: data.get('feature') for node, data in graph.nodes(data=True)}
                feature_df = pd.DataFrame(
                    {f"{node[0]}:{node[1][1]}": feat for node, feat in node_features.items() if feat is not None}
                ).T


                feature_names = [
                    'A', 'U', 'G', 'C', 'N',
                    'C3_endo', 'C2_endo',
                    'stem', 'hairpin', 'internal_loop', 'bulge', 'multi_branch',
                    'accessibility',

                    'alpha_cos', 'alpha_sin',
                    'beta_cos', 'beta_sin',
                    'gamma_cos', 'gamma_sin',
                    'delta_cos', 'delta_sin',
                    'epsilon_cos', 'epsilon_sin',
                    'zeta_cos', 'zeta_sin',
                    'chi_cos', 'chi_sin',
                    'pseudo_cos', 'pseudo_sin',
                    'is_paired',
                    'is_purine', 'is_pyrimidine',
                    'has_phosphate'
                ]

                if feature_df.shape[1] == len(feature_names):
                    feature_df.columns = feature_names

                feature_df.to_csv(os.path.join(output_dir, f"{pdb_id}_node_features.csv"))

            except Exception as e:
                print(f"Error processing {pdb_id}: {e}")
                pbar.update(1)
                continue

            pbar.update(1)


    with open(os.path.join(output_dir, 'rna_structures_data.pkl'), 'wb') as f:

        serializable_results = {}
        for pdb_id, data in results.items():
            serializable_results[pdb_id] = {
                'contact_map': data['contact_map'],
                'residue_info': data['residue_info'],
                'pdb_path': data['pdb_path']

            }
        pickle.dump(serializable_results, f)

    return results

def analyze_rna_features(results):
    'Analyze rna features.'
    print("\nRNA Feature Analysis Summary:")
    print("=" * 50)

    all_features = []
    feature_info = []


    for pdb_id, data in results.items():
        graph = data['graph']
        for node, node_data in graph.nodes(data=True):
            if 'feature' in node_data and node_data['feature'] is not None:
                feat = node_data['feature']
                all_features.append(feat)
                feature_info.append({
                    'pdb_id': pdb_id,
                    'node': node,
                    'dim': len(feat)
                })


    dims = [info['dim'] for info in feature_info]
    unique_dims = set(dims)
    print(f"Observed feature dimensions: {unique_dims}")

    if len(unique_dims) > 1:
        print("\nFeatures with unexpected dimensions:")
        most_common_dim = max(set(dims), key=dims.count)
        print(f"Most common dimension: {most_common_dim}")


        abnormal_info = [info for info in feature_info if info['dim'] != most_common_dim]
        for i, info in enumerate(abnormal_info[:5]):
            print(f"  {i + 1}. PDB: {info['pdb_id']}, node: {info['node']}, dimension: {info['dim']}")

        if len(abnormal_info) > 5:
            print(f"  ... and {len(abnormal_info) - 5} additional mismatches")


    most_common_dim = max(set(dims), key=dims.count)
    filtered_features = [feat for i, feat in enumerate(all_features) if dims[i] == most_common_dim]

    print(f"\nFeatures retained: {len(filtered_features)}/{len(all_features)}")


    if filtered_features:
        features_array = np.vstack(filtered_features)


        feature_means = np.mean(features_array, axis=0)
        feature_stds = np.std(features_array, axis=0)
        feature_mins = np.min(features_array, axis=0)
        feature_maxs = np.max(features_array, axis=0)


        feature_names = [
            'A', 'U', 'G', 'C', 'N',  # 5
            'C3_endo', 'C2_endo',  # 2
            'stem', 'hairpin', 'internal_loop', 'bulge', 'multi_branch',  # 5
            'accessibility',  # 1
            'alpha_cos', 'alpha_sin',  # 2
            'beta_cos', 'beta_sin',  # 2
            'gamma_cos', 'gamma_sin',  # 2
            'delta_cos', 'delta_sin',  # 2
            'epsilon_cos', 'epsilon_sin',  # 2
            'zeta_cos', 'zeta_sin',  # 2
            'chi_cos', 'chi_sin',  # 2
            'pseudo_cos', 'pseudo_sin',  # 2
            'is_paired',  # 1
            'is_purine', 'is_pyrimidine',  # 2
            'has_phosphate'  # 1
        ]


        if len(feature_names) > most_common_dim:
            feature_names = feature_names[:most_common_dim]
        elif len(feature_names) < most_common_dim:
            for i in range(len(feature_names), most_common_dim):
                feature_names.append(f'feature_{i}')


        stats_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean': feature_means,
            'Std': feature_stds,
            'Min': feature_mins,
            'Max': feature_maxs
        })

        print("\nFeature Statistics:")
        print(stats_df)


        output_dir = './rna_analysis'
        os.makedirs(output_dir, exist_ok=True)
        stats_df.to_csv(os.path.join(output_dir, 'rna_feature_stats.csv'), index=False)

        return stats_df
    else:
        print("No valid features found for analysis.")
        return None

def rna_contact_map(structure, threshold=8.0):
    'Rna contact map.'

    residues = []
    residue_info = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in list(rna_dict.keys()):
                    residues.append(residue)
                    residue_info.append((chain.id, residue.id))

    n_residues = len(residues)
    contact_map = np.zeros((n_residues, n_residues))


    for i in range(n_residues):
        for j in range(n_residues):
            if i != j:

                min_distance = float('inf')

                for atom1 in residues[i]:
                    for atom2 in residues[j]:
                        distance = np.linalg.norm(atom1.coord - atom2.coord)
                        if distance < min_distance:
                            min_distance = distance


                if min_distance <= threshold:
                    contact_map[i, j] = 1

    return contact_map, residue_info


if __name__ == "__main__":

    directory = "./processed_pdbs/nucleic_chains"
    output_dir = "./rna_analysis"


    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        sys.exit(1)


    os.makedirs(output_dir, exist_ok=True)


    results = process_rna_pdbs(directory, output_dir)


    stats_df = analyze_rna_features(results)

    print("\nRNA processing completed successfully!")
    print(f"Results saved to {output_dir}")
