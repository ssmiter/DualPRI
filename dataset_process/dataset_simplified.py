import os
import numpy as np
import pickle
import pandas as pd
import networkx as nx
from torch.ao.nn.quantized.functional import threshold
from tqdm import tqdm
from Bio import PDB
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import logging
import warnings


warnings.filterwarnings("ignore", category=PDB.PDBExceptions.PDBConstructionWarning)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("improved_interface_builder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


from utils import (create_graph_from_structure, get_residue_ca_coords,
                   positional_encoding, load_conservation_scores, get_rna_center_coords)
from enhanced_rna_processing import create_rna_graph_from_structure




def calculate_contact_maps(protein_coords, rna_coords, thresholds=[8.0, 12.0, 16.0]):
    'Calculate contact maps.'
    contact_maps = {}


    dist_matrix = cdist(protein_coords, rna_coords)


    for threshold in thresholds:
        contact_maps[threshold] = (dist_matrix < threshold)

    return contact_maps, dist_matrix


def identify_multiscale_interface(protein_coords, rna_coords, thresholds=[8.0, 12.0, 16.0]):
    'Identify multiscale interface.'
    if len(protein_coords) == 0 or len(rna_coords) == 0:
        return {
            'pairs': [],
            'protein_mask': np.zeros(len(protein_coords), dtype=bool),
            'rna_mask': np.zeros(len(rna_coords), dtype=bool),
            'contact_counts': {},
            'contact_maps': {}
        }


    contact_maps, dist_matrix = calculate_contact_maps(protein_coords, rna_coords, thresholds)


    all_pairs = []

    for threshold in thresholds:
        contacts = contact_maps[threshold]
        p_indices, r_indices = np.where(contacts)

        for p_idx, r_idx in zip(p_indices, r_indices):

            pair_key = (p_idx, r_idx)
            if not any(p['protein_idx'] == p_idx and p['rna_idx'] == r_idx for p in all_pairs):
                all_pairs.append({
                    'protein_idx': p_idx,
                    'rna_idx': r_idx,
                    'distance': dist_matrix[p_idx, r_idx],
                    'threshold': threshold
                })


    protein_mask = np.zeros(len(protein_coords), dtype=bool)
    rna_mask = np.zeros(len(rna_coords), dtype=bool)

    for pair in all_pairs:
        protein_mask[pair['protein_idx']] = True
        rna_mask[pair['rna_idx']] = True


    contact_counts = {threshold: np.sum(contact_maps[threshold]) for threshold in thresholds}

    return {
        'pairs': all_pairs,
        'protein_mask': protein_mask,
        'rna_mask': rna_mask,
        'contact_counts': contact_counts,
        'contact_maps': contact_maps,
        'dist_matrix': dist_matrix
    }


def create_segmented_contact_features(protein_coords, rna_coords,
                                                thresholds=[8.0, 10.0, 15.0, 20.0, 30.0, 50.0], use_torch=True):
    'Create segmented contact features.'
    import numpy as np


    num_classes = len(thresholds) + 1
    num_protein = len(protein_coords)
    num_rna = len(rna_coords)


    if num_protein == 0 or num_rna == 0:
        protein_contact_features = np.zeros((num_protein, num_classes))
        rna_contact_features = np.zeros((num_rna, num_classes))
        protein_contact_intensity = np.zeros((num_protein, 1))
        rna_contact_intensity = np.zeros((num_rna, 1))
        return protein_contact_features, rna_contact_features, protein_contact_intensity, rna_contact_intensity


    if use_torch:
        import torch


        if not isinstance(protein_coords, torch.Tensor):
            protein_coords_tensor = torch.tensor(protein_coords, dtype=torch.float32)
        else:
            protein_coords_tensor = protein_coords.float()

        if not isinstance(rna_coords, torch.Tensor):
            rna_coords_tensor = torch.tensor(rna_coords, dtype=torch.float32)
        else:
            rna_coords_tensor = rna_coords.float()


        dist_matrix = torch.cdist(protein_coords_tensor, rna_coords_tensor)


        protein_contact_features = torch.zeros((num_protein, num_classes))
        rna_contact_features = torch.zeros((num_rna, num_classes))


        protein_min_dist, _ = torch.min(dist_matrix, dim=1)
        protein_intensity = 1.0 / (1.0 + torch.exp((protein_min_dist - 8.0) / 2.0))
        protein_contact_intensity = protein_intensity.unsqueeze(1)


        rna_min_dist, _ = torch.min(dist_matrix, dim=0)
        rna_intensity = 1.0 / (1.0 + torch.exp((rna_min_dist - 8.0) / 2.0))
        rna_contact_intensity = rna_intensity.unsqueeze(1)



        mask = (dist_matrix < thresholds[0])
        protein_contact_features[:, 0] = mask.sum(dim=1).float() / num_rna


        for j in range(1, len(thresholds)):
            mask = (dist_matrix >= thresholds[j - 1]) & (dist_matrix < thresholds[j])
            protein_contact_features[:, j] = mask.sum(dim=1).float() / num_rna


        mask = (dist_matrix >= thresholds[-1])
        protein_contact_features[:, -1] = mask.sum(dim=1).float() / num_rna



        mask = (dist_matrix < thresholds[0])
        rna_contact_features[:, 0] = mask.sum(dim=0).float() / num_protein


        for j in range(1, len(thresholds)):
            mask = (dist_matrix >= thresholds[j - 1]) & (dist_matrix < thresholds[j])
            rna_contact_features[:, j] = mask.sum(dim=0).float() / num_protein


        mask = (dist_matrix >= thresholds[-1])
        rna_contact_features[:, -1] = mask.sum(dim=0).float() / num_protein


        protein_contact_features = protein_contact_features.numpy()
        rna_contact_features = rna_contact_features.numpy()
        protein_contact_intensity = protein_contact_intensity.numpy()
        rna_contact_intensity = rna_contact_intensity.numpy()


    else:
        from scipy.spatial.distance import cdist


        protein_contact_features = np.zeros((num_protein, num_classes))
        rna_contact_features = np.zeros((num_rna, num_classes))
        protein_contact_intensity = np.zeros((num_protein, 1))
        rna_contact_intensity = np.zeros((num_rna, 1))


        dist_matrix = cdist(protein_coords, rna_coords)


        for i in range(num_protein):
            distances = dist_matrix[i]


            min_dist = np.min(distances)
            intensity = 1.0 / (1.0 + np.exp((min_dist - 8.0) / 2.0))
            protein_contact_intensity[i, 0] = intensity



            mask = (distances < thresholds[0])
            protein_contact_features[i, 0] = np.sum(mask) / num_rna


            for j in range(1, len(thresholds)):
                mask = (distances >= thresholds[j - 1]) & (distances < thresholds[j])
                protein_contact_features[i, j] = np.sum(mask) / num_rna


            mask = (distances >= thresholds[-1])
            protein_contact_features[i, -1] = np.sum(mask) / num_rna


        for i in range(num_rna):
            distances = dist_matrix[:, i]


            min_dist = np.min(distances)
            intensity = 1.0 / (1.0 + np.exp((min_dist - 8.0) / 2.0))
            rna_contact_intensity[i, 0] = intensity



            mask = (distances < thresholds[0])
            rna_contact_features[i, 0] = np.sum(mask) / num_protein


            for j in range(1, len(thresholds)):
                mask = (distances >= thresholds[j - 1]) & (distances < thresholds[j])
                rna_contact_features[i, j] = np.sum(mask) / num_protein


            mask = (distances >= thresholds[-1])
            rna_contact_features[i, -1] = np.sum(mask) / num_protein

    return protein_contact_features, rna_contact_features, protein_contact_intensity, rna_contact_intensity


def create_multiscale_contact_features(protein_coords, rna_coords, thresholds=[8.0, 12.0, 16.0]):
    'Create multiscale contact features.'

    protein_contact_features = np.zeros((len(protein_coords), len(thresholds)))
    rna_contact_features = np.zeros((len(rna_coords), len(thresholds)))


    if len(protein_coords) == 0 or len(rna_coords) == 0:
        return protein_contact_features, rna_contact_features


    dist_matrix = cdist(protein_coords, rna_coords)


    for i, threshold in enumerate(thresholds):

        contact_mask = (dist_matrix < threshold)


        protein_contact_features[:, i] = contact_mask.sum(axis=1) / len(rna_coords)


        rna_contact_features[:, i] = contact_mask.sum(axis=0) / len(protein_coords)

    return protein_contact_features, rna_contact_features


def assign_chemical_edge_weights(protein_features, rna_features, interface_pairs):
    'Assign chemical edge weights.'

    # 0:ALA, 1:ARG, 2:ASN, 3:ASP, 4:CYS, 5:GLN, 6:GLU, 7:GLY, 8:HIS, 9:ILE,
    # 10:LEU, 11:LYS, 12:MET, 13:PHE, 14:PRO, 15:SER, 16:THR, 17:TRP, 18:TYR, 19:VAL




    interaction_matrix = np.ones((20, 5)) * 0.8


    interaction_matrix[1, 2] = 1.5


    interaction_matrix[11, :] = 1.2


    interaction_matrix[8, 0] = 1.3


    interaction_matrix[3, 3] = 1.2


    interaction_matrix[6, 3] = 1.2

    edge_weights = []
    edge_features = []

    for pair in interface_pairs:
        p_idx = pair['protein_idx']
        r_idx = pair['rna_idx']
        distance = pair['distance']


        p_type = np.argmax(protein_features[p_idx][:20]) if p_idx < len(protein_features) else 0
        r_type = np.argmax(rna_features[r_idx][:5]) if r_idx < len(rna_features) else 0


        base_weight = 1.0 / (1.0 + distance / 5.0)


        chemical_weight = interaction_matrix[p_type, r_type]


        weight = base_weight * chemical_weight


        pos_encoding = positional_encoding(distance)



        protein_type = np.zeros(20)
        protein_type[p_type] = 1


        rna_type = np.zeros(5)
        rna_type[r_type] = 1


        combined_feat = np.concatenate([
            pos_encoding,
            [weight],
            [distance],
            [chemical_weight],
            protein_type.astype(np.float32),
            rna_type.astype(np.float32)
        ])

        edge_weights.append(weight)
        edge_features.append(combined_feat)

    return np.array(edge_weights), np.array(edge_features)


def construct_interface_graph(protein_graph, rna_graph, interface_info, edge_offset=0):
    'Construct interface graph.'
    interface_graph = nx.Graph()


    protein_features = protein_graph.features
    protein_coords = protein_graph.coords
    rna_features = rna_graph.features
    rna_coords = rna_graph.coords


    pairs = interface_info['pairs']


    edge_weights, edge_features = assign_chemical_edge_weights(
        protein_features, rna_features, pairs
    )


    for i, pair in enumerate(pairs):
        p_idx = pair['protein_idx']
        r_idx = pair['rna_idx']
        r_idx_global = r_idx + len(protein_features)


        interface_graph.add_edge(
            p_idx,
            r_idx_global,
            weight=edge_weights[i],
            feature=edge_features[i],
            distance=pair['distance'],
            threshold=pair['threshold']
        )

    return interface_graph


def create_unified_multimodal_graph(wild_protein_graph, mutant_protein_graph, rna_graph,
                                    wild_interface, mutant_interface, mutation_pos):
    'Create unified multimodal graph.'

    wild_interface_graph = construct_interface_graph(
        wild_protein_graph, rna_graph, wild_interface
    )

    mutant_interface_graph = construct_interface_graph(
        mutant_protein_graph, rna_graph, mutant_interface
    )


    n_wild = len(wild_protein_graph.features)
    n_mutant = len(mutant_protein_graph.features)
    n_rna = len(rna_graph.features)


    max_feat_dim = max(
        wild_protein_graph.features.shape[1],
        mutant_protein_graph.features.shape[1],
        rna_graph.features.shape[1]
    )


    def pad_features(features, target_dim):
        if features.shape[1] < target_dim:
            padding = np.zeros((features.shape[0], target_dim - features.shape[1]))
            return np.hstack([features, padding])
        return features

    wild_features = pad_features(wild_protein_graph.features, max_feat_dim)
    mutant_features = pad_features(mutant_protein_graph.features, max_feat_dim)
    rna_features = pad_features(rna_graph.features, max_feat_dim)


    all_features = np.vstack([
        wild_features,
        mutant_features,
        rna_features
    ])


    all_coords = np.vstack([
        wild_protein_graph.coords,
        mutant_protein_graph.coords,
        rna_graph.coords
    ])

    global_center = np.mean(all_coords, axis=0)

    wild_coords_rel = wild_protein_graph.coords - global_center
    mutant_coords_rel = mutant_protein_graph.coords - global_center
    rna_coords_rel = rna_graph.coords - global_center

    all_coords_rel = np.vstack([
        wild_coords_rel,
        mutant_coords_rel,
        rna_coords_rel
    ])


    wild_mask = np.zeros(n_wild + n_mutant + n_rna, dtype=bool)
    wild_mask[:n_wild] = True

    mutant_mask = np.zeros(n_wild + n_mutant + n_rna, dtype=bool)
    mutant_mask[n_wild:n_wild + n_mutant] = True

    rna_mask = np.zeros(n_wild + n_mutant + n_rna, dtype=bool)
    rna_mask[n_wild + n_mutant:] = True


    wild_interface_mask = np.zeros(n_wild + n_mutant + n_rna, dtype=bool)
    wild_interface_mask[:n_wild][wild_interface['protein_mask']] = True
    wild_interface_mask[n_wild + n_mutant:][wild_interface['rna_mask']] = True

    mutant_interface_mask = np.zeros(n_wild + n_mutant + n_rna, dtype=bool)
    mutant_interface_mask[n_wild:n_wild + n_mutant][mutant_interface['protein_mask']] = True
    mutant_interface_mask[n_wild + n_mutant:][mutant_interface['rna_mask']] = True

    interface_mask = wild_interface_mask | mutant_interface_mask




    wild_edge_index = wild_protein_graph.edge_index.copy()
    mutant_edge_index = mutant_protein_graph.edge_index.copy() + n_wild
    rna_edge_index = rna_graph.edge_index.copy() + (n_wild + n_mutant)


    wild_interface_edge_index = []
    wild_interface_edge_attr = []

    for u, v, data in wild_interface_graph.edges(data=True):
        if v >= n_wild:
            v = v - n_wild + (n_wild + n_mutant)
        wild_interface_edge_index.append([u, v])
        wild_interface_edge_index.append([v, u])

        edge_feat = data.get('feature', np.zeros(44))
        wild_interface_edge_attr.append(edge_feat)
        wild_interface_edge_attr.append(edge_feat)

    mutant_interface_edge_index = []
    mutant_interface_edge_attr = []

    for u, v, data in mutant_interface_graph.edges(data=True):
        u = u + n_wild
        if v >= n_mutant:
            v = v - n_mutant + (n_wild + n_mutant)
        mutant_interface_edge_index.append([u, v])
        mutant_interface_edge_index.append([v, u])

        edge_feat = data.get('feature', np.zeros(44))
        mutant_interface_edge_attr.append(edge_feat)
        mutant_interface_edge_attr.append(edge_feat)


    all_edge_indices = []
    all_edge_attrs = []
    all_edge_types = []


    if len(wild_edge_index) > 0:
        all_edge_indices.append(wild_edge_index)
        all_edge_attrs.append(wild_protein_graph.edge_attr)
        all_edge_types.extend([0] * wild_edge_index.shape[1])


    if len(mutant_edge_index) > 0:
        all_edge_indices.append(mutant_edge_index)
        all_edge_attrs.append(mutant_protein_graph.edge_attr)
        all_edge_types.extend([1] * mutant_edge_index.shape[1])


    if len(rna_edge_index) > 0:
        all_edge_indices.append(rna_edge_index)
        all_edge_attrs.append(rna_graph.edge_attr)
        all_edge_types.extend([2] * rna_edge_index.shape[1])


    if wild_interface_edge_index:
        all_edge_indices.append(np.array(wild_interface_edge_index).T)
        all_edge_attrs.append(np.array(wild_interface_edge_attr))
        all_edge_types.extend([3] * len(wild_interface_edge_index))


    if mutant_interface_edge_index:
        all_edge_indices.append(np.array(mutant_interface_edge_index).T)
        all_edge_attrs.append(np.array(mutant_interface_edge_attr))
        all_edge_types.extend([4] * len(mutant_interface_edge_index))


    edge_index = np.hstack(all_edge_indices) if all_edge_indices else np.zeros((2, 0), dtype=np.int64)



    max_edge_feat_dim = max([attr.shape[1] for attr in all_edge_attrs]) if all_edge_attrs else 0


    padded_edge_attrs = []
    for attr in all_edge_attrs:
        if attr.shape[1] < max_edge_feat_dim:
            padding = np.zeros((attr.shape[0], max_edge_feat_dim - attr.shape[1]))
            padded_edge_attrs.append(np.hstack([attr, padding]))
        else:
            padded_edge_attrs.append(attr)

    edge_attr = np.vstack(padded_edge_attrs) if padded_edge_attrs else np.zeros((0, max_edge_feat_dim),
                                                                                dtype=np.float32)
    edge_type = np.array(all_edge_types, dtype=np.int32)



    if mutation_pos is not None:
        global_mutation_pos = mutation_pos
        mutant_mutation_pos = mutation_pos + n_wild
    else:
        global_mutation_pos = None
        mutant_mutation_pos = None


    unified_graph = {
        'node_features': all_features,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'edge_type': edge_type,
        'pos': all_coords_rel,
        'global_center': global_center,
        'wild_mask': wild_mask,
        'mutant_mask': mutant_mask,
        'rna_mask': rna_mask,
        'interface_mask': interface_mask,
        'wild_interface_mask': wild_interface_mask,
        'mutant_interface_mask': mutant_interface_mask,
        'mutation_index': global_mutation_pos,
        'mutant_mutation_index': mutant_mutation_pos,
        'feature_dims': {
            'protein': wild_protein_graph.features.shape[1],
            'rna': rna_graph.features.shape[1],
            'unified': max_feat_dim,
            'edge': max_edge_feat_dim
        }
    }

    return unified_graph


def find_mutation_position(wild_features, mutant_features):
    'Find mutation position.'
    for i in range(len(wild_features)):

        if not np.array_equal(wild_features[i][:20], mutant_features[i][:20]):
            wild_aa = np.argmax(wild_features[i][:20])
            mutant_aa = np.argmax(mutant_features[i][:20])
            return i, {
                'wild_aa': wild_aa,
                'mutant_aa': mutant_aa,
                'position': i
            }
    return None, None


def build_enhanced_multimodal_dataset(csv_file, protein_dir, rna_dir, pssm_file, conservation_dir,
                                      output_file, thresholds=[8.0],
                                      max_file_size_mb=1.0):
    'Build enhanced multimodal dataset.'
    logger.info("Building the multimodal dataset with segmented contact features.")


    with open(pssm_file, 'rb') as f:
        pssm_dict = pickle.load(f)


    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} mutation records.")


    os.makedirs(os.path.dirname(output_file), exist_ok=True)


    parser = PDB.PDBParser(QUIET=True)


    dataset = []
    skipped = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing mutations"):
        pdb_id = row['PDB Id']
        chain_id = row['Mutated Chain']
        mutation = row['Mutation_PDB']
        ddg = row['DDGexp']
        mutation_pos = row.get('Mutation_Position', None)


        wild_protein_path = os.path.join(protein_dir, f"{pdb_id}_{chain_id}_wild_type.pdb")
        mutant_protein_path = os.path.join(protein_dir, f"{pdb_id}_{chain_id}_mutant_{mutation}.pdb")
        rna_path = os.path.join(rna_dir, f"{pdb_id}_nucleic.pdb")


        if not all(os.path.exists(path) for path in [wild_protein_path, mutant_protein_path, rna_path]):
            skipped.append((pdb_id, chain_id, mutation, "missing file"))
            continue


        file_sizes_mb = {
            'wild': os.path.getsize(wild_protein_path) / (1024 * 1024),
            'mutant': os.path.getsize(mutant_protein_path) / (1024 * 1024),
            'rna': os.path.getsize(rna_path) / (1024 * 1024)
        }

        if any(size > max_file_size_mb for size in file_sizes_mb.values()):
            too_large = [name for name, size in file_sizes_mb.items() if size > max_file_size_mb]
            logger.warning(
                f"Skipping {pdb_id}_{chain_id}_{mutation}: {','.join(too_large)} exceeds {max_file_size_mb} MB")
            skipped.append((pdb_id, chain_id, mutation, f"file exceeds {max_file_size_mb} MB"))
            continue

        try:

            wild_structure = parser.get_structure("wild", wild_protein_path)
            mutant_structure = parser.get_structure("mutant", mutant_protein_path)
            rna_structure = parser.get_structure("rna", rna_path)


            wild_pssm = pssm_dict.get(f"{pdb_id}_{chain_id}_wild_type.pssm", None)
            wild_conservation = load_conservation_scores(conservation_dir, f"{pdb_id}_{chain_id}_wild_type.pdb")

            mutant_pssm = pssm_dict.get(f"{pdb_id}_{chain_id}_mutant_{mutation}.pssm", None)
            mutant_conservation = load_conservation_scores(conservation_dir,
                                                           f"{pdb_id}_{chain_id}_mutant_{mutation}.pdb")


            wild_protein_graph = create_graph_from_structure(wild_structure, wild_pssm, wild_conservation)
            mutant_protein_graph = create_graph_from_structure(mutant_structure, mutant_pssm, mutant_conservation)
            rna_graph = create_rna_graph_from_structure(rna_structure, threshold=8.0)


            wild_ca_coords = get_residue_ca_coords(wild_structure)
            mutant_ca_coords = get_residue_ca_coords(mutant_structure)
            rna_center_coords = get_rna_center_coords(rna_structure)


            wild_ca_coords_array = np.array(list(wild_ca_coords.values()))
            mutant_ca_coords_array = np.array(list(mutant_ca_coords.values()))
            rna_center_coords_array = np.array(list(rna_center_coords.values()))


            wild_interface = identify_multiscale_interface(
                wild_ca_coords_array, rna_center_coords_array, thresholds
            )

            mutant_interface = identify_multiscale_interface(
                mutant_ca_coords_array, rna_center_coords_array, thresholds
            )


            wild_protein_features = np.array([data.get('feature', np.zeros(41))
                                              for _, data in wild_protein_graph.nodes(data=True)])
            mutant_protein_features = np.array([data.get('feature', np.zeros(41))
                                                for _, data in mutant_protein_graph.nodes(data=True)])


            if mutation_pos is None:
                mutation_pos, mutation_details = find_mutation_position(wild_protein_features, mutant_protein_features)
            else:

                if wild_protein_features[mutation_pos][:20].any() and mutant_protein_features[mutation_pos][:20].any():
                    wild_aa = np.argmax(wild_protein_features[mutation_pos][:20])
                    mutant_aa = np.argmax(mutant_protein_features[mutation_pos][:20])
                    mutation_details = {
                        'wild_aa': wild_aa,
                        'mutant_aa': mutant_aa,
                        'position': mutation_pos
                    }
                else:
                    mutation_details = None


            rna_nodes = list(rna_graph.nodes())
            rna_onehot_features = np.zeros((len(rna_nodes), 5))

            for i, node in enumerate(rna_nodes):
                data = rna_graph.nodes[node]
                if 'feature' in data and len(data['feature']) >= 5:

                    rna_onehot_features[i, :] = data['feature'][:5]


            wild_edges = []
            wild_edge_attr = []

            for u, v, data in wild_protein_graph.edges(data=True):

                u_idx = list(wild_protein_graph.nodes()).index(u)
                v_idx = list(wild_protein_graph.nodes()).index(v)

                wild_edges.append([u_idx, v_idx])
                wild_edges.append([v_idx, u_idx])

                edge_feat = data.get('feature', np.zeros(16))
                wild_edge_attr.append(edge_feat)
                wild_edge_attr.append(edge_feat)

            mutant_edges = []
            mutant_edge_attr = []

            for u, v, data in mutant_protein_graph.edges(data=True):
                u_idx = list(mutant_protein_graph.nodes()).index(u)
                v_idx = list(mutant_protein_graph.nodes()).index(v)

                mutant_edges.append([u_idx, v_idx])
                mutant_edges.append([v_idx, u_idx])

                edge_feat = data.get('feature', np.zeros(16))
                mutant_edge_attr.append(edge_feat)
                mutant_edge_attr.append(edge_feat)

            rna_edges = []
            rna_edge_attr = []

            for u, v, data in rna_graph.edges(data=True):
                u_idx = list(rna_graph.nodes()).index(u)
                v_idx = list(rna_graph.nodes()).index(v)

                rna_edges.append([u_idx, v_idx])
                rna_edges.append([v_idx, u_idx])

                edge_feat = data.get('feature', np.zeros(16))
                rna_edge_attr.append(edge_feat)
                rna_edge_attr.append(edge_feat)


            wild_edge_index = np.array(wild_edges).T if wild_edges else np.zeros((2, 0), dtype=np.int64)
            wild_edge_attr = np.array(wild_edge_attr) if wild_edge_attr else np.zeros((0, 16), dtype=np.float32)

            mutant_edge_index = np.array(mutant_edges).T if mutant_edges else np.zeros((2, 0), dtype=np.int64)
            mutant_edge_attr = np.array(mutant_edge_attr) if mutant_edge_attr else np.zeros((0, 16), dtype=np.float32)

            rna_edge_index = np.array(rna_edges).T if rna_edges else np.zeros((2, 0), dtype=np.int64)
            rna_edge_attr = np.array(rna_edge_attr) if rna_edge_attr else np.zeros((0, 16), dtype=np.float32)

            # =================================================================================

            # =================================================================================

            # wild_contact_features, rna_contact_features_wild, wild_contact_intensity, rna_contact_intensity_wild = create_segmented_contact_features(
            #     wild_ca_coords_array, rna_center_coords_array, thresholds
            # )
            #
            # mutant_contact_features, rna_contact_features_mutant, mutant_contact_intensity, rna_contact_intensity_mutant = create_segmented_contact_features(
            #     mutant_ca_coords_array, rna_center_coords_array, thresholds
            # )
            #

            # rna_contact_features = (rna_contact_features_wild + rna_contact_features_mutant) / 2.0
            # rna_contact_intensity = (rna_contact_intensity_wild + rna_contact_intensity_mutant) / 2.0


            # wild_cumulative_features, rna_cumulative_features_wild = create_multiscale_contact_features(
            #     wild_ca_coords_array, rna_center_coords_array, thresholds
            # )
            #
            # mutant_cumulative_features, rna_cumulative_features_mutant = create_multiscale_contact_features(
            #     mutant_ca_coords_array, rna_center_coords_array, thresholds
            # )
            #

            # rna_cumulative_features = (rna_cumulative_features_wild + rna_cumulative_features_mutant) / 2.0


            # wild_protein_features = np.concatenate([



            # ], axis=1)
            #
            # mutant_protein_features = np.concatenate([



            # ], axis=1)
            #

            rna_features = rna_onehot_features
            # rna_features = np.concatenate([



            # ], axis=1)
            # =================================================================================


            feature_dims = {
                'base_dim': 41,



            }


            create_unified_graph = False


            unified_graph = create_unified_multimodal_graph(
                wild_protein_graph, mutant_protein_graph, rna_graph,
                wild_interface, mutant_interface, mutation_pos
            ) if create_unified_graph else None


            sample = {
                'unified_graph': unified_graph,
                'wild_graph': {
                    'features': wild_protein_features,
                    'coords': wild_ca_coords_array,
                    'edge_index': wild_edge_index,
                    'edge_attr': wild_edge_attr,
                    'interface': wild_interface
                },
                'mutant_graph': {
                    'features': mutant_protein_features,
                    'coords': mutant_ca_coords_array,
                    'edge_index': mutant_edge_index,
                    'edge_attr': mutant_edge_attr,
                    'interface': mutant_interface
                },
                'rna_graph': {
                    'features': rna_features,
                    'coords': rna_center_coords_array,
                    'edge_index': rna_edge_index,
                    'edge_attr': rna_edge_attr
                },
                'metadata': {
                    'pdb_id': pdb_id,
                    'chain_id': chain_id,
                    'mutation': mutation,
                    'ddg': ddg,
                    'mutation_pos': mutation_pos,
                    'mutation_details': mutation_details,
                    'feature_dims': feature_dims,
                    'thresholds': thresholds,

                }
            }

            dataset.append(sample)


            if len(dataset) % 50 == 0:
                logger.info(f"Processed {len(dataset)} samples; skipped {len(skipped)}.")


                avg_wild_contacts = np.mean([len(s['wild_graph']['interface']['pairs']) for s in dataset])
                avg_mutant_contacts = np.mean([len(s['mutant_graph']['interface']['pairs']) for s in dataset])

                logger.info(f"Mean wild-type interface contacts: {avg_wild_contacts:.2f}")
                logger.info(f"Mean mutant interface contacts: {avg_mutant_contacts:.2f}")

        except Exception as e:
            logger.error(f"Could not process {pdb_id}_{chain_id}_{mutation}: {e}")
            skipped.append((pdb_id, chain_id, mutation, str(e)))
            continue


    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)


    with open(os.path.join(os.path.dirname(output_file), "skipped_samples.txt"), 'w') as f:
        for item in skipped:
            f.write(f"{item[0]}_{item[1]}_{item[2]}: {item[3]}\n")


    logger.info(f"Built {len(dataset)} samples and saved them to {output_file}")
    logger.info(f"Skipped {len(skipped)} samples; see skipped_samples.txt")

    return dataset






def build_multimodal_dataset_simplified(csv_file, protein_dir, rna_dir, pssm_file, conservation_dir,
                                        output_file, thresholds=[8.0, 12.0, 16.0], max_file_size_mb=1.0):
    'Build multimodal dataset simplified.'
    logger.info("Building the compact multimodal dataset with RNA one-hot and contact features.")


    with open(pssm_file, 'rb') as f:
        pssm_dict = pickle.load(f)


    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} mutation records.")


    os.makedirs(os.path.dirname(output_file), exist_ok=True)


    parser = PDB.PDBParser(QUIET=True)


    dataset = []
    skipped = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing mutations"):
        pdb_id = row['PDB Id']
        chain_id = row['Mutated Chain']
        mutation = row['Mutation_PDB']
        ddg = row['DDGexp']


        wild_protein_path = os.path.join(protein_dir, f"{pdb_id}_{chain_id}_wild_type.pdb")
        mutant_protein_path = os.path.join(protein_dir, f"{pdb_id}_{chain_id}_mutant_{mutation}.pdb")
        rna_path = os.path.join(rna_dir, f"{pdb_id}_nucleic.pdb")


        if not all(os.path.exists(path) for path in [wild_protein_path, mutant_protein_path, rna_path]):
            skipped.append((pdb_id, chain_id, mutation, "missing file"))
            continue


        file_sizes_mb = {
            'wild': os.path.getsize(wild_protein_path) / (1024 * 1024),
            'mutant': os.path.getsize(mutant_protein_path) / (1024 * 1024),
            'rna': os.path.getsize(rna_path) / (1024 * 1024)
        }

        if any(size > max_file_size_mb for size in file_sizes_mb.values()):
            too_large = [name for name, size in file_sizes_mb.items() if size > max_file_size_mb]
            logger.warning(
                f"Skipping {pdb_id}_{chain_id}_{mutation}: {','.join(too_large)} exceeds {max_file_size_mb} MB")
            skipped.append((pdb_id, chain_id, mutation, f"file exceeds {max_file_size_mb} MB"))
            continue

        try:

            wild_structure = parser.get_structure("wild", wild_protein_path)
            mutant_structure = parser.get_structure("mutant", mutant_protein_path)
            rna_structure = parser.get_structure("rna", rna_path)


            wild_pssm = pssm_dict.get(f"{pdb_id}_{chain_id}_wild_type.pssm", None)
            wild_conservation = load_conservation_scores(conservation_dir, f"{pdb_id}_{chain_id}_wild_type.pdb")

            mutant_pssm = pssm_dict.get(f"{pdb_id}_{chain_id}_mutant_{mutation}.pssm", None)
            mutant_conservation = load_conservation_scores(conservation_dir,
                                                           f"{pdb_id}_{chain_id}_mutant_{mutation}.pdb")


            wild_protein_graph = create_graph_from_structure(wild_structure, wild_pssm, wild_conservation)
            mutant_protein_graph = create_graph_from_structure(mutant_structure, mutant_pssm, mutant_conservation)
            rna_graph = create_rna_graph_from_structure(rna_structure, threshold=8.0)


            wild_ca_coords = get_residue_ca_coords(wild_structure)
            mutant_ca_coords = get_residue_ca_coords(mutant_structure)
            rna_center_coords = get_rna_center_coords(rna_structure)


            wild_ca_coords_array = np.array(list(wild_ca_coords.values()))
            mutant_ca_coords_array = np.array(list(mutant_ca_coords.values()))
            rna_center_coords_array = np.array(list(rna_center_coords.values()))


            wild_interface = identify_multiscale_interface(
                wild_ca_coords_array, rna_center_coords_array, thresholds
            )

            mutant_interface = identify_multiscale_interface(
                mutant_ca_coords_array, rna_center_coords_array, thresholds
            )


            mutation_pos = None
            for i, (node1, node2) in enumerate(zip(wild_protein_graph.nodes(), mutant_protein_graph.nodes())):
                data1 = wild_protein_graph.nodes[node1]
                data2 = mutant_protein_graph.nodes[node2]

                if 'feature' in data1 and 'feature' in data2:
                    feat1 = data1['feature']
                    feat2 = data2['feature']

                    if len(feat1) >= 20 and len(feat2) >= 20 and not np.array_equal(feat1[:20], feat2[:20]):
                        mutation_pos = i
                        break


            wild_protein_features = np.array([data.get('feature', np.zeros(41))
                                              for _, data in wild_protein_graph.nodes(data=True)])
            mutant_protein_features = np.array([data.get('feature', np.zeros(41))
                                                for _, data in mutant_protein_graph.nodes(data=True)])


            rna_nodes = list(rna_graph.nodes())
            rna_onehot_features = np.zeros((len(rna_nodes), 5))

            for i, node in enumerate(rna_nodes):
                data = rna_graph.nodes[node]
                if 'feature' in data and len(data['feature']) >= 5:

                    rna_onehot_features[i, :] = data['feature'][:5]


            wild_edges = []
            wild_edge_attr = []

            for u, v, data in wild_protein_graph.edges(data=True):

                u_idx = list(wild_protein_graph.nodes()).index(u)
                v_idx = list(wild_protein_graph.nodes()).index(v)

                wild_edges.append([u_idx, v_idx])
                wild_edges.append([v_idx, u_idx])

                edge_feat = data.get('feature', np.zeros(16))
                wild_edge_attr.append(edge_feat)
                wild_edge_attr.append(edge_feat)

            mutant_edges = []
            mutant_edge_attr = []

            for u, v, data in mutant_protein_graph.edges(data=True):
                u_idx = list(mutant_protein_graph.nodes()).index(u)
                v_idx = list(mutant_protein_graph.nodes()).index(v)

                mutant_edges.append([u_idx, v_idx])
                mutant_edges.append([v_idx, u_idx])

                edge_feat = data.get('feature', np.zeros(16))
                mutant_edge_attr.append(edge_feat)
                mutant_edge_attr.append(edge_feat)

            rna_edges = []
            rna_edge_attr = []

            for u, v, data in rna_graph.edges(data=True):
                u_idx = list(rna_graph.nodes()).index(u)
                v_idx = list(rna_graph.nodes()).index(v)

                rna_edges.append([u_idx, v_idx])
                rna_edges.append([v_idx, u_idx])

                edge_feat = data.get('feature', np.zeros(16))
                rna_edge_attr.append(edge_feat)
                rna_edge_attr.append(edge_feat)


            wild_edge_index = np.array(wild_edges).T if wild_edges else np.zeros((2, 0), dtype=np.int64)
            wild_edge_attr = np.array(wild_edge_attr) if wild_edge_attr else np.zeros((0, 16), dtype=np.float32)

            mutant_edge_index = np.array(mutant_edges).T if mutant_edges else np.zeros((2, 0), dtype=np.int64)
            mutant_edge_attr = np.array(mutant_edge_attr) if mutant_edge_attr else np.zeros((0, 16), dtype=np.float32)

            rna_edge_index = np.array(rna_edges).T if rna_edges else np.zeros((2, 0), dtype=np.int64)
            rna_edge_attr = np.array(rna_edge_attr) if rna_edge_attr else np.zeros((0, 16), dtype=np.float32)


            wild_contact_features, rna_contact_features_wild = create_multiscale_contact_features(
                wild_ca_coords_array, rna_center_coords_array, thresholds
            )

            mutant_contact_features, rna_contact_features_mutant = create_multiscale_contact_features(
                mutant_ca_coords_array, rna_center_coords_array, thresholds
            )


            rna_contact_features = (rna_contact_features_wild + rna_contact_features_mutant) / 2.0


            wild_protein_features = np.concatenate([wild_protein_features, wild_contact_features], axis=1)
            mutant_protein_features = np.concatenate([mutant_protein_features, mutant_contact_features], axis=1)


            rna_features = np.concatenate([rna_onehot_features, rna_contact_features], axis=1)


            unified_graph = None


            sample = {
                'unified_graph': unified_graph,
                'wild_graph': {
                    'features': wild_protein_features,
                    'coords': wild_ca_coords_array,
                    'edge_index': wild_edge_index,
                    'edge_attr': wild_edge_attr,
                    'interface': wild_interface
                },
                'mutant_graph': {
                    'features': mutant_protein_features,
                    'coords': mutant_ca_coords_array,
                    'edge_index': mutant_edge_index,
                    'edge_attr': mutant_edge_attr,
                    'interface': mutant_interface
                },
                'rna_graph': {
                    'features': rna_features,
                    'coords': rna_center_coords_array,
                    'edge_index': rna_edge_index,
                    'edge_attr': rna_edge_attr
                },
                'metadata': {
                    'pdb_id': pdb_id,
                    'chain_id': chain_id,
                    'mutation': mutation,
                    'ddg': ddg,
                    'mutation_pos': mutation_pos,
                    'contact_feature_dim': len(thresholds),
                    'rna_simplified': True
                }
            }

            dataset.append(sample)


            if len(dataset) % 50 == 0:
                logger.info(f"Processed {len(dataset)} samples; skipped {len(skipped)}.")


                avg_wild_contacts = np.mean([len(s['wild_graph']['interface']['pairs']) for s in dataset])
                avg_mutant_contacts = np.mean([len(s['mutant_graph']['interface']['pairs']) for s in dataset])

                logger.info(f"Mean wild-type interface contacts: {avg_wild_contacts:.2f}")
                logger.info(f"Mean mutant interface contacts: {avg_mutant_contacts:.2f}")

        except Exception as e:
            logger.error(f"Could not process {pdb_id}_{chain_id}_{mutation}: {e}")
            skipped.append((pdb_id, chain_id, mutation, str(e)))
            continue


    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)


    with open(os.path.join(os.path.dirname(output_file), "skipped_samples.txt"), 'w') as f:
        for item in skipped:
            f.write(f"{item[0]}_{item[1]}_{item[2]}: {item[3]}\n")

    logger.info(f"Built {len(dataset)} samples and saved them to {output_file}")
    logger.info(f"Skipped {len(skipped)} samples; see skipped_samples.txt")
    logger.info(f"RNA feature dimension: {rna_features.shape[1]} (one-hot=5, contact={len(thresholds)})")

    return dataset


def build_multimodal_dataset(csv_file, protein_dir, rna_dir, pssm_file, conservation_dir,
                             output_file, thresholds=[8.0, 12.0, 16.0], max_file_size_mb=1.0):
    'Build multimodal dataset.'

    return build_multimodal_dataset_simplified(
        csv_file, protein_dir, rna_dir, pssm_file, conservation_dir,
        output_file, thresholds, max_file_size_mb
    )




def visualize_interface_contacts(sample, output_dir="interface_viz"):
    'Visualize interface contacts.'
    os.makedirs(output_dir, exist_ok=True)

    pdb_id = sample['metadata']['pdb_id']
    chain_id = sample['metadata']['chain_id']
    mutation = sample['metadata']['mutation']
    ddg = sample['metadata']['ddg']
    mutation_pos = sample['metadata']['mutation_pos']


    wild_interface = sample['wild_graph']['interface']
    mutant_interface = sample['mutant_graph']['interface']

    wild_coords = sample['wild_graph']['coords']
    mutant_coords = sample['mutant_graph']['coords']
    rna_coords = sample['rna_graph']['coords']


    fig, axes = plt.subplots(1, 2, figsize=(15, 7))


    wild_contact = np.zeros((len(wild_coords), len(rna_coords)))
    for pair in wild_interface['pairs']:
        wild_contact[pair['protein_idx'], pair['rna_idx']] = 1

    axes[0].imshow(wild_contact, cmap='viridis')
    axes[0].set_title(f'Wild Type Interface Contacts\n{pdb_id}_{chain_id} - {len(wild_interface["pairs"])} contacts')
    axes[0].set_xlabel('RNA Residues')
    axes[0].set_ylabel('Protein Residues')


    if mutation_pos is not None:
        axes[0].axhline(y=mutation_pos, color='r', linestyle='--', alpha=0.5)


    mutant_contact = np.zeros((len(mutant_coords), len(rna_coords)))
    for pair in mutant_interface['pairs']:
        mutant_contact[pair['protein_idx'], pair['rna_idx']] = 1

    axes[1].imshow(mutant_contact, cmap='viridis')
    axes[1].set_title(
        f'Mutant Type Interface Contacts\n{mutation} (DDG: {ddg:.2f}) - {len(mutant_interface["pairs"])} contacts')
    axes[1].set_xlabel('RNA Residues')
    axes[1].set_ylabel('Protein Residues')


    if mutation_pos is not None:
        axes[1].axhline(y=mutation_pos, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{pdb_id}_{chain_id}_{mutation}_interface.png"), dpi=300)
    plt.close()


    diff_contact = mutant_contact - wild_contact

    plt.figure(figsize=(10, 8))
    plt.imshow(diff_contact, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Mutant - Wild')
    plt.title(f'Interface Contact Differences\n{pdb_id}_{chain_id} {mutation} (DDG: {ddg:.2f})')
    plt.xlabel('RNA Residues')
    plt.ylabel('Protein Residues')


    if mutation_pos is not None:
        plt.axhline(y=mutation_pos, color='black', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{pdb_id}_{chain_id}_{mutation}_interface_diff.png"), dpi=300)
    plt.close()


    contact_dim = sample['metadata'].get('contact_feature_dim', 0)
    if contact_dim > 0:

        wild_features = sample['wild_graph']['features']
        mutant_features = sample['mutant_graph']['features']

        wild_contact_features = wild_features[:, -contact_dim:]
        mutant_contact_features = mutant_features[:, -contact_dim:]


        fig, axes = plt.subplots(1, 2, figsize=(15, 7))


        im0 = axes[0].imshow(wild_contact_features, cmap='viridis', aspect='auto')
        axes[0].set_title(f'Wild Type Contact Features\n{pdb_id}_{chain_id}')
        axes[0].set_xlabel('Threshold Levels')
        axes[0].set_ylabel('Protein Residues')
        fig.colorbar(im0, ax=axes[0])


        im1 = axes[1].imshow(mutant_contact_features, cmap='viridis', aspect='auto')
        axes[1].set_title(f'Mutant Type Contact Features\n{mutation} (DDG: {ddg:.2f})')
        axes[1].set_xlabel('Threshold Levels')
        axes[1].set_ylabel('Protein Residues')
        fig.colorbar(im1, ax=axes[1])


        if mutation_pos is not None:
            axes[0].axhline(y=mutation_pos, color='r', linestyle='--', alpha=0.5)
            axes[1].axhline(y=mutation_pos, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{pdb_id}_{chain_id}_{mutation}_contact_features.png"), dpi=300)
        plt.close()


        diff_features = mutant_contact_features - wild_contact_features
        plt.figure(figsize=(10, 8))
        im = plt.imshow(diff_features, cmap='coolwarm', aspect='auto')
        plt.colorbar(im, label='Mutant - Wild')
        plt.title(f'Contact Feature Differences\n{pdb_id}_{chain_id} {mutation} (DDG: {ddg:.2f})')
        plt.xlabel('Threshold Levels')
        plt.ylabel('Protein Residues')


        if mutation_pos is not None:
            plt.axhline(y=mutation_pos, color='black', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{pdb_id}_{chain_id}_{mutation}_contact_features_diff.png"), dpi=300)
        plt.close()


def analyze_dataset(dataset, output_dir="dataset_analysis"):
    'Analyze dataset.'
    os.makedirs(output_dir, exist_ok=True)


    n_samples = len(dataset)
    pdb_ids = set([s['metadata']['pdb_id'] for s in dataset])
    chains = set([s['metadata']['chain_id'] for s in dataset])


    protein_nodes = [len(s['wild_graph']['features']) for s in dataset]
    rna_nodes = [len(s['rna_graph']['features']) for s in dataset]


    protein_feat_dims = set([s['wild_graph']['features'].shape[1] for s in dataset])
    rna_feat_dims = set([s['rna_graph']['features'].shape[1] for s in dataset])


    has_unified = all(s['unified_graph'] is not None for s in dataset)
    unified_feat_dims = set() if not has_unified else set(
        [s['unified_graph']['node_features'].shape[1] for s in dataset])


    edge_counts = [] if not has_unified else [s['unified_graph']['edge_index'].shape[1] for s in dataset]


    edge_types = []
    if has_unified:
        for s in dataset:
            edge_types.extend(s['unified_graph']['edge_type'].tolist())

    from collections import defaultdict
    type_counts = defaultdict(int)
    for t in edge_types:
        type_counts[t] += 1


    ddg_values = [s['metadata']['ddg'] for s in dataset]


    wild_interface_counts = [len(s['wild_graph']['interface']['pairs']) for s in dataset]
    mutant_interface_counts = [len(s['mutant_graph']['interface']['pairs']) for s in dataset]


    has_contact_features = 'contact_feature_dim' in dataset[0]['metadata']
    contact_dim = dataset[0]['metadata'].get('contact_feature_dim', 0) if has_contact_features else 0


    plt.figure(figsize=(10, 6))
    plt.hist(ddg_values, bins=20, alpha=0.7)
    plt.axvline(x=np.mean(ddg_values), color='r', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(ddg_values):.2f}')
    plt.title('DDG distribution')
    plt.xlabel('DDG')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "ddg_distribution.png"), dpi=300)
    plt.close()


    plt.figure(figsize=(10, 6))
    plt.hist(wild_interface_counts, bins=20, alpha=0.5, label='Wild Type')
    plt.hist(mutant_interface_counts, bins=20, alpha=0.5, label='Mutant')
    plt.title('Interface-contact count distribution')
    plt.xlabel('Contact count')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "interface_contacts_distribution.png"), dpi=300)
    plt.close()


    if has_contact_features and contact_dim > 0:

        all_wild_contacts = np.vstack([s['wild_graph']['features'][:, -contact_dim:].mean(axis=0) for s in dataset])
        all_mutant_contacts = np.vstack([s['mutant_graph']['features'][:, -contact_dim:].mean(axis=0) for s in dataset])


        plt.figure(figsize=(10, 6))
        thresholds = [f'T{i + 1}' for i in range(contact_dim)]
        plt.bar(np.arange(contact_dim) - 0.2, all_wild_contacts.mean(axis=0), width=0.4, alpha=0.7, label='Wild Type')
        plt.bar(np.arange(contact_dim) + 0.2, all_mutant_contacts.mean(axis=0), width=0.4, alpha=0.7, label='Mutant')
        plt.xticks(np.arange(contact_dim), thresholds)
        plt.title('Mean contact-feature distribution')
        plt.xlabel('Threshold index')
        plt.ylabel('Mean contact proportion')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "contact_features_distribution.png"), dpi=300)
        plt.close()


    logger.info("Dataset summary:")
    logger.info(f"Samples: {n_samples}")
    logger.info(f"Unique PDB entries: {len(pdb_ids)}")
    logger.info(f"Unique chains: {len(chains)}")
    logger.info(f"Mean protein nodes: {np.mean(protein_nodes):.2f}")
    logger.info(f"Mean RNA nodes: {np.mean(rna_nodes):.2f}")
    logger.info(f"Protein feature dimensions: {protein_feat_dims}")
    logger.info(f"RNA feature dimensions: {rna_feat_dims}")
    if has_unified:
        logger.info(f"Unified graph feature dimensions: {unified_feat_dims}")
        logger.info(f"Mean edges: {np.mean(edge_counts):.2f}")
        logger.info(f"Edge-type distribution: {type_counts}")
    logger.info(f"DDG range: [{min(ddg_values):.2f}, {max(ddg_values):.2f}]")
    logger.info(f"Mean DDG: {np.mean(ddg_values):.2f}")
    logger.info(f"DDG standard deviation: {np.std(ddg_values):.2f}")
    logger.info(f"Mean wild-type interface pairs: {np.mean(wild_interface_counts):.2f}")
    if has_contact_features:
        logger.info(f"Contact feature dimension: {contact_dim}")


    visualize_samples = [dataset[i] for i in np.linspace(0, len(dataset) - 1, 5, dtype=int)]
    for sample in visualize_samples:
        visualize_interface_contacts(sample, output_dir)


    stats = {
        'n_samples': n_samples,
        'pdb_ids': pdb_ids,
        'chains': chains,
        'protein_nodes': protein_nodes,
        'rna_nodes': rna_nodes,
        'protein_feat_dims': protein_feat_dims,
        'rna_feat_dims': rna_feat_dims,
        'unified_feat_dims': unified_feat_dims if has_unified else None,
        'edge_counts': edge_counts if has_unified else None,
        'type_counts': type_counts if has_unified else None,
        'ddg_values': ddg_values,
        'wild_interface_counts': wild_interface_counts,
        'mutant_interface_counts': mutant_interface_counts,
        'has_contact_features': has_contact_features,
        'contact_feature_dim': contact_dim
    }

    with open(os.path.join(output_dir, "dataset_statistics.pkl"), 'wb') as f:
        pickle.dump(stats, f)

    logger.info(f"Visualizations saved to {output_dir}")
    logger.info(f"Dataset statistics saved to {os.path.join(output_dir, 'dataset_statistics.pkl')}")

    return stats




def analyze_sample(dataset, sample_idx=0):
    'Analyze sample.'
    sample = dataset[sample_idx]

    logger.info("\n==================================================")
    logger.info(f"Detailed analysis of sample {sample_idx}")
    logger.info("==================================================")


    metadata = sample['metadata']
    logger.info(f"PDB ID: {metadata['pdb_id']}")
    logger.info(f"Chain: {metadata['chain_id']}")
    logger.info(f"Mutation: {metadata['mutation']}")
    logger.info(f"DDG: {metadata['ddg']}")
    logger.info(f"Mutation position: {metadata['mutation_pos']}")


    logger.info("\nWild-type graph:")
    logger.info(f"Protein nodes: {len(sample['wild_graph']['features'])}")
    logger.info(f"Protein feature dimension: {sample['wild_graph']['features'].shape[1]}")
    logger.info(f"RNA nodes: {len(sample['rna_graph']['features'])}")
    logger.info(f"RNA feature dimension: {sample['rna_graph']['features'].shape[1]}")
    logger.info(f"Interface pairs: {len(sample['wild_graph']['interface']['pairs'])}")


    logger.info("\nMutant graph:")
    logger.info(f"Protein nodes: {len(sample['mutant_graph']['features'])}")
    logger.info(f"Interface pairs: {len(sample['mutant_graph']['interface']['pairs'])}")


    if 'contact_feature_dim' in metadata:
        contact_dim = metadata['contact_feature_dim']
        logger.info("\nContact features:")
        logger.info(f"Contact feature dimension: {contact_dim}")


        wild_contact_features = sample['wild_graph']['features'][:, -contact_dim:]
        mutant_contact_features = sample['mutant_graph']['features'][:, -contact_dim:]

        logger.info(f"Mean wild-type contact features: {wild_contact_features.mean(axis=0)}")
        logger.info(f"Mean mutant contact features: {mutant_contact_features.mean(axis=0)}")


        logger.info(f"Mean contact-feature difference: {(mutant_contact_features - wild_contact_features).mean(axis=0)}")


    if sample['unified_graph'] is not None:
        unified = sample['unified_graph']
        logger.info("\nUnified graph:")
        logger.info(f"Nodes: {unified['node_features'].shape[0]}")
        logger.info(f"Feature dimension: {unified['node_features'].shape[1]}")
        logger.info(f"Edges: {unified['edge_index'].shape[1]}")


        edge_type_counts = {}
        for t in range(5):
            edge_type_counts[t] = np.sum(unified['edge_type'] == t)

        logger.info("\nEdge-type distribution:")
        logger.info(f"Wild-type protein: {edge_type_counts.get(0, 0)}")
        logger.info(f"Mutant protein: {edge_type_counts.get(1, 0)}")
        logger.info(f"RNA: {edge_type_counts.get(2, 0)}")
        logger.info(f"Wild-type interface: {edge_type_counts.get(3, 0)}")
        logger.info(f"Mutant interface: {edge_type_counts.get(4, 0)}")


        logger.info("\nFeature dimensions:")
        logger.info(f"Protein: {unified['feature_dims']['protein']}")
        logger.info(f"RNA: {unified['feature_dims']['rna']}")
        logger.info(f"Unified graph: {unified['feature_dims']['unified']}")

    return sample




def main():
    'Main.'

    COMPLEX_DIR = "./Dataset/S394"
    PROTEIN_DIR = "./processed_pdbs/protein_chains"
    RNA_DIR = "./processed_pdbs/nucleic_chains"
    PSSM_FILE = "./Dataset/PSSM_394/pssm_s394.pkl"
    CONSERVATION_DIR = "./Dataset/cons_s394"
    CSV_FILE = "./Dataset/S394.csv"


    OUTPUT_DIR = "./dataset"


    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "enhanced_protein_rna_dataset_with_contacts.pkl")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # dataset1 = build_multimodal_dataset_enhanced(
    #     csv_file=CSV_FILE,
    #     protein_dir=PROTEIN_DIR,
    #     rna_dir=RNA_DIR,
    #     pssm_file=PSSM_FILE,
    #     conservation_dir=CONSERVATION_DIR,
    #     output_file=OUTPUT_FILE,
    #     thresholds=[8.0, 12.0, 16.0],
    #     max_file_size_mb=1.0
    # )


    # OUTPUT_FILE_SIMPLIFIED = os.path.join(OUTPUT_DIR, "simplified_protein_rna_dataset.pkl")
    # dataset = build_enhanced_multimodal_dataset(
    #     csv_file=CSV_FILE,
    #     protein_dir=PROTEIN_DIR,
    #     rna_dir=RNA_DIR,
    #     pssm_file=PSSM_FILE,
    #     conservation_dir=CONSERVATION_DIR,
    #     output_file=OUTPUT_FILE_SIMPLIFIED,
    #     thresholds=[7.0, 8.0],
    #     max_file_size_mb=1.0
    # )


    OUTPUT_FILE_ENHANCED = os.path.join(OUTPUT_DIR, "protein_rna_dataset.pkl")
    dataset = build_enhanced_multimodal_dataset(
        csv_file=CSV_FILE,
        protein_dir=PROTEIN_DIR,
        rna_dir=RNA_DIR,
        pssm_file=PSSM_FILE,
        conservation_dir=CONSERVATION_DIR,
        output_file=OUTPUT_FILE_ENHANCED,
        # thresholds=DEFAULT_CONTACT_THRESHOLDS,  # deprecated
        max_file_size_mb=5.0
    )


    # analyze_dataset(dataset1, output_dir="dataset_analysis_enhanced")
    analyze_dataset(dataset, output_dir="dataset_analysis_enhanced")


    sample_indices = [0, len(dataset) // 4, len(dataset) // 2, 3 * len(dataset) // 4, len(dataset) - 1]
    for idx in sample_indices:
        analyze_sample(dataset, idx)


if __name__ == "__main__":
    main()
