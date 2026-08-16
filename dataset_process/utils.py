import logging
import os
import sys
from pathlib import Path

import numpy as np
import networkx as nx
from Bio import PDB
import pickle
import re
import pandas as pd
from tqdm import tqdm

from config import ENCODING_DIM

def get_residue_ca_coords(structure):
    ca_coords = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_coords[(chain.id, residue.id)] = residue['CA'].coord
    return ca_coords



aa_dict = {
    'ALA': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ARG': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ASN': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ASP': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'CYS': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'GLN': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'GLU': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'GLY': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'HIS': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ILE': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'LEU': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'LYS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'MET': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'PHE': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'PRO': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'SER': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'THR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'TRP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'TYR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'VAL': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}


def get_residue_ca_coords(structure):
    ca_coords = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_coords[(chain.id, residue.id)] = residue['CA'].coord
    return ca_coords


def positional_encoding(distance, dimension=16):
    encoding = np.zeros(dimension)
    for i in range(dimension):
        encoding[i] = np.sin(distance / (10000 ** ((2 * i) / dimension))) if i % 2 == 0 else np.cos(
            distance / (10000 ** ((2 * i) / dimension)))
    return encoding


def load_conservation_scores(directory, filename):
    conservation_file = os.path.join(directory, filename.replace('.pdb', '_conservation.npy'))
    if os.path.exists(conservation_file):
        return np.load(conservation_file)
    return None


def create_graph_from_structure(structure, pssm_scores, conservation_scores, threshold=8.0, encoding_dim=ENCODING_DIM):
    ca_coords = get_residue_ca_coords(structure)
    graph = nx.Graph()

    # Add nodes with combined features
    for i, ((chain_id, residue_id), coord) in enumerate(ca_coords.items()):
        residue = structure[0][chain_id][residue_id]
        resname = residue.get_resname()
        one_hot = np.array(aa_dict.get(resname, [0] * 20))  # Convert one-hot encoding to a NumPy array
        pssm = np.array(pssm_scores[i]) if pssm_scores is not None and i < len(pssm_scores) else np.zeros(20)

        pssm = np.clip(pssm, -10, 10)
        pssm = (pssm + 10) / 20
        conservation = np.array([conservation_scores[i]]) if conservation_scores is not None and i < len(
            conservation_scores) else np.zeros(1)
        features = np.concatenate((one_hot, pssm, conservation))  # Concatenate features
        graph.add_node((chain_id, residue_id), feature=features)

    # Add edges based on distance threshold
    for (chain_id1, res_id1), coord1 in ca_coords.items():
        for (chain_id2, res_id2), coord2 in ca_coords.items():
            if (chain_id1, res_id1) != (chain_id2, res_id2):
                distance = np.linalg.norm(coord1 - coord2)
                if distance < threshold:
                    edge_feature = positional_encoding(distance, encoding_dim)
                    graph.add_edge((chain_id1, res_id1), (chain_id2, res_id2), feature=edge_feature)

    return graph


def process_pdb_files(directory, pssm_pickle_file, conservation_directory, progress_bar=True):
    parser = PDB.PDBParser(QUIET=True)
    graphs = []

    wild_type_files = {}
    mutant_files = {}

    # Load PSSM scores
    with open(pssm_pickle_file, 'rb') as file:
        pssm_scores_dict = pickle.load(file)

    # Classify files as wild-type or mutant
    for file_name in os.listdir(directory):
        if re.match(r'.+_wild_type\.pdb', file_name):
            wild_type_files[file_name.split('_wild_type')[0]] = file_name
        elif re.match(r'.+_mutant_.+\.pdb', file_name):
            base_name = file_name.split('_mutant_')[0]
            if base_name not in mutant_files:
                mutant_files[base_name] = []
            mutant_files[base_name].append(file_name)


    wild_pdbs = list(Path(directory).glob('*_wild_type.pdb'))

    total_mutants = sum(len(list(wild_pdb.parent.glob(f'{wild_pdb.stem.replace("_wild_type", "")}_mutant_*.pdb')))
                        for wild_pdb in wild_pdbs)

    if progress_bar:
        pbar = tqdm(total=total_mutants,
                    desc="Processing mutations",
                    unit="mutation",
                    ncols=100,
                    colour='green',
                    file=sys.stdout)
    # with tqdm(total=total_mutants,
    #           desc="Processing mutations",
    #           unit="mutation",
    #           ncols=100,
    #           colour='green',
    #           file=sys.stdout) as pbar:
        # Process each mutant file with its corresponding wild type
    for base_name, mutants in mutant_files.items():
        if base_name in wild_type_files:
            wild_type_file = wild_type_files[base_name]
            wild_type_structure = parser.get_structure(base_name + '_wild_type',
                                                       os.path.join(directory, wild_type_file))

            # Load PSSM and conservation scores for wild type
            wild_type_pssm = pssm_scores_dict.get(wild_type_file.replace('.pdb', '.pssm'), None)
            wild_type_conservation = load_conservation_scores(conservation_directory, wild_type_file)
            wild_type_graph = create_graph_from_structure(wild_type_structure, wild_type_pssm, wild_type_conservation)

            for mutant_file in mutants:
                mutant_structure = parser.get_structure(mutant_file, os.path.join(directory, mutant_file))

                # Load PSSM and conservation scores for mutant
                mutant_pssm = pssm_scores_dict.get(mutant_file.replace('.pdb', '.pssm'), None)
                mutant_conservation = load_conservation_scores(conservation_directory, mutant_file)

                mutant_graph = create_graph_from_structure(mutant_structure, mutant_pssm, mutant_conservation)

                graphs.append((wild_type_graph, mutant_graph, mutant_file))
                if progress_bar:
                    pbar.update(1)


    if progress_bar:
        pbar.close()

    return graphs

# -----------------------------------------------RNA-----------------------------------------------



rna_dict = {
    'A': 'ADE',
    'U': 'URI',
    'G': 'GUA',
    'C': 'CYT',
}


rna_one_hot = {
    'ADE': [1, 0, 0, 0],  # A
    'URI': [0, 1, 0, 0],  # U
    'GUA': [0, 0, 1, 0],  # G
    'CYT': [0, 0, 0, 1],  # C
}


def get_rna_center_coords_v1(structure):
    'Get rna center coords v1.'
    center_coords = {}
    for model in structure:
        for chain in model:
            for residue in chain:

                if residue.get_resname() in ['ADE', 'URI', 'GUA', 'CYT', 'A', 'U', 'G', 'C']:

                    if residue.get_resname() in ['ADE', 'GUA', 'A', 'G']:
                        if 'N9' in residue:
                            center_coords[(chain.id, residue.id)] = residue['N9'].coord

                    elif residue.get_resname() in ['CYT', 'URI', 'C', 'U']:
                        if 'N1' in residue:
                            center_coords[(chain.id, residue.id)] = residue['N1'].coord

                    if (chain.id, residue.id) not in center_coords and "C1'" in residue:
                        center_coords[(chain.id, residue.id)] = residue["C1'"].coord
    return center_coords


def get_rna_center_coords_v2(structure):
    'Get rna center coords v2.'
    from enhanced_rna_processing import rna_dict, purine_types, pyrimidine_types

    center_coords = {}

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in list(rna_dict.keys()):
                    res_id = (chain.id, residue.id)


                    resname = residue.get_resname()
                    if resname in purine_types:

                        if 'N9' in residue:
                            center_coords[res_id] = residue['N9'].coord
                        elif "C1'" in residue:
                            center_coords[res_id] = residue["C1'"].coord
                        else:

                            atoms = [atom.coord for atom in residue]
                            if atoms:
                                center_coords[res_id] = np.mean(atoms, axis=0)
                    elif resname in pyrimidine_types:

                        if 'N1' in residue:
                            center_coords[res_id] = residue['N1'].coord
                        elif "C1'" in residue:
                            center_coords[res_id] = residue["C1'"].coord
                        else:

                            atoms = [atom.coord for atom in residue]
                            if atoms:
                                center_coords[res_id] = np.mean(atoms, axis=0)
                    else:

                        atoms = [atom.coord for atom in residue]
                        if atoms:
                            center_coords[res_id] = np.mean(atoms, axis=0)

    return center_coords


def get_rna_center_coords(structure):
    'Get rna center coords.'

    from enhanced_rna_processing import rna_dict

    center_coords = {}

    for model in structure:
        for chain in model:
            for residue in chain:

                if residue.get_resname() in list(rna_dict.keys()):
                    res_id = (chain.id, residue.id)


                    atoms = [atom.coord for atom in residue]
                    if atoms:
                        center_coord = np.mean(atoms, axis=0)
                        center_coords[res_id] = center_coord

    return center_coords

def rna_contact_map(structure, threshold=8.0):
    'Rna contact map.'
    center_coords = get_rna_center_coords(structure)
    residue_info = []


    for (chain_id, res_id) in center_coords.keys():
        residue = structure[0][chain_id][res_id]
        residue_info.append((chain_id, res_id, residue.get_resname()))

    n_residues = len(residue_info)
    contact_map = np.zeros((n_residues, n_residues))


    coords_list = list(center_coords.values())
    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            distance = np.linalg.norm(coords_list[i] - coords_list[j])
            if distance <= threshold:
                contact_map[i, j] = 1
                contact_map[j, i] = 1

    return contact_map, residue_info


def create_rna_graph_from_structure(structure, threshold=8.0, encoding_dim=ENCODING_DIM):
    'Create rna graph from structure.'
    center_coords = get_rna_center_coords(structure)
    graph = nx.Graph()


    for (chain_id, residue_id), coord in center_coords.items():
        residue = structure[0][chain_id][residue_id]
        resname = residue.get_resname()

        if resname in ['A', 'U', 'G', 'C']:
            resname = rna_dict.get(resname, resname)

        one_hot = np.array(rna_one_hot.get(resname, [0, 0, 0, 0]))

        graph.add_node((chain_id, residue_id),
                       feature=one_hot,
                       resname=resname)


    for (chain_id1, res_id1), coord1 in center_coords.items():
        for (chain_id2, res_id2), coord2 in center_coords.items():
            if (chain_id1, res_id1) != (chain_id2, res_id2):
                distance = np.linalg.norm(coord1 - coord2)
                if distance < threshold:
                    edge_feature = positional_encoding(distance, encoding_dim)
                    graph.add_edge((chain_id1, res_id1),
                                   (chain_id2, res_id2),
                                   feature=edge_feature,
                                   distance=distance)

    return graph
