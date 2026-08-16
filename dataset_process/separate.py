#!/usr/bin/env python3
import os
import subprocess
import glob
import re
from collections import defaultdict


AMINO_ACIDS = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

NUCLEOTIDES = {
    'A': 'A', 'G': 'G', 'C': 'C', 'U': 'U', 'T': 'T',
    'DA': 'A', 'DG': 'G', 'DC': 'C', 'DT': 'T',
    'ADE': 'A', 'GUA': 'G', 'CYT': 'C', 'THY': 'T', 'URA': 'U',

    'RAD': 'A', 'RG': 'G', 'RC': 'C', 'RU': 'U'
}


PDB_DIR = "./Dataset/S394_pdbs"
OUTPUT_DIR = "./processed_pdbs"


def create_tcl_script():
    'Create tcl script.'

    os.makedirs(f"{OUTPUT_DIR}/protein_chains", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/nucleic_chains", exist_ok=True)

    script_content = f"""
# Create output directories.
file mkdir {OUTPUT_DIR}/protein_chains
file mkdir {OUTPUT_DIR}/nucleic_chains

# Process every PDB file in the selected directory.
foreach pdbfile [glob {PDB_DIR}/*.pdb] {{
    # Extract the PDB identifier without its path or extension.
    set pdbid [file rootname [file tail $pdbfile]]
    puts "Processing $pdbid..."

    # Load the structure.
    mol new $pdbfile

    # Separate protein and nucleic-acid chains.
    set protein [atomselect top "protein"]
    set nucleic [atomselect top "nucleic"]

    # Write the separated structures.
    $protein writepdb {OUTPUT_DIR}/protein_chains/${{pdbid}}_protein.pdb
    $nucleic writepdb {OUTPUT_DIR}/nucleic_chains/${{pdbid}}_nucleic.pdb

    # Release the loaded molecule before the next input file.
    $protein delete
    $nucleic delete
    mol delete top
}}

puts "All PDB files processed."
# Exit VMD after batch processing.
quit
"""
    with open(f"{OUTPUT_DIR}/separate_all_chains.tcl", "w") as f:
        f.write(script_content)
    print(f"Tcl script created: {OUTPUT_DIR}/separate_all_chains.tcl")


def run_vmd_script():
    'Run vmd script.'
    try:
        print("Running the VMD batch script...")
        result = subprocess.run(
            ["vmd", "-dispdev", "text", "-e", f"{OUTPUT_DIR}/separate_all_chains.tcl"],
            check=True,
            text=True,
            capture_output=True
        )
        print("VMD processing complete.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"VMD failed: {e}")
        print(f"VMD error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("VMD was not found. Install it and make sure it is available on PATH.")
        return False


def extract_protein_sequence(pdb_file):
    'Extract protein sequence.'
    if not os.path.exists(pdb_file):
        print(f"File not found: {pdb_file}")
        return {}


    chains = defaultdict(dict)


    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                chain_id = line[21]
                res_num = int(line[22:26])
                res_name = line[17:20].strip()

                if res_name in AMINO_ACIDS:
                    chains[chain_id][res_num] = AMINO_ACIDS[res_name]


    sequences = {}
    for chain_id, residues in chains.items():

        sorted_residues = [residues[k] for k in sorted(residues.keys())]
        sequences[chain_id] = ''.join(sorted_residues)

    return sequences


def extract_rna_sequence(pdb_file):
    'Extract rna sequence.'
    if not os.path.exists(pdb_file):
        print(f"File not found: {pdb_file}")
        return {}


    chains = defaultdict(dict)


    processed_residues = set()


    with open(pdb_file, 'r') as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            chain_id = line[21]
            res_num = int(line[22:26])
            res_name = line[17:20].strip()


            res_key = (chain_id, res_num)
            if res_key in processed_residues:
                continue


            base = None
            if res_name in NUCLEOTIDES:
                base = NUCLEOTIDES[res_name]
                processed_residues.add(res_key)
                chains[chain_id][res_num] = base


    sequences = {}
    for chain_id, residues in chains.items():

        sorted_residues = [residues[k] for k in sorted(residues.keys())]
        sequences[chain_id] = ''.join(sorted_residues)

        if 'T' in sequences[chain_id]:
            print(f"Note: chain {chain_id} may be DNA rather than RNA because it contains thymine.")

    return sequences


def save_sequence_to_fasta(pdb_id, sequences, output_file):
    'Save sequence to fasta.'
    with open(output_file, 'w') as f:
        for chain_id, sequence in sequences.items():
            if sequence:
                f.write(f">{pdb_id}_{chain_id}\n")

                for i in range(0, len(sequence), 80):
                    f.write(f"{sequence[i:i + 80]}\n")


def append_to_fasta(pdb_id, sequences, output_file):
    'Append to fasta.'

    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            pass


    with open(output_file, 'a') as f:
        for chain_id, sequence in sequences.items():
            if sequence:
                f.write(f">{pdb_id}_{chain_id}\n")

                for i in range(0, len(sequence), 80):
                    f.write(f"{sequence[i:i + 80]}\n")


def process_all_pdbs(create_summary=True):
    'Process all pdbs.'

    os.makedirs(f"{OUTPUT_DIR}/sequences", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/sequences/protein", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/sequences/nucleic", exist_ok=True)


    protein_files = glob.glob(f"{OUTPUT_DIR}/protein_chains/*.pdb")
    nucleic_files = glob.glob(f"{OUTPUT_DIR}/nucleic_chains/*.pdb")


    if create_summary:
        summary_protein = f"{OUTPUT_DIR}/sequences/all_protein_sequences.fasta"
        summary_nucleic = f"{OUTPUT_DIR}/sequences/all_nucleic_sequences.fasta"

        open(summary_protein, 'w').close()
        open(summary_nucleic, 'w').close()

    protein_count = 0
    nucleic_count = 0


    for pdb_file in protein_files:

        pdb_id = os.path.basename(pdb_file).split('_')[0]
        print(f"Extracting protein sequence: {pdb_id}")

        sequences = extract_protein_sequence(pdb_file)
        if sequences:
            protein_count += 1

            output_file = f"{OUTPUT_DIR}/sequences/protein/{pdb_id}_protein.fasta"
            save_sequence_to_fasta(pdb_id, sequences, output_file)


            if create_summary:
                append_to_fasta(pdb_id, sequences, summary_protein)


    for pdb_file in nucleic_files:

        pdb_id = os.path.basename(pdb_file).split('_')[0]
        print(f"Extracting nucleic-acid sequence: {pdb_id}")

        sequences = extract_rna_sequence(pdb_file)
        if sequences:
            nucleic_count += 1

            output_file = f"{OUTPUT_DIR}/sequences/nucleic/{pdb_id}_nucleic.fasta"
            save_sequence_to_fasta(pdb_id, sequences, output_file)


            if create_summary:
                append_to_fasta(pdb_id, sequences, summary_nucleic)

    print(f"Extracted {protein_count} protein and {nucleic_count} nucleic-acid sequence files.")
    print("Individual sequence files were saved to:")
    print(f"- {OUTPUT_DIR}/sequences/protein/")
    print(f"- {OUTPUT_DIR}/sequences/nucleic/")

    if create_summary:
        print("Combined sequence files were saved to:")
        print(f"- {summary_protein}")
        print(f"- {summary_nucleic}")


def main():
    'Main.'
    print("=== PDB chain separation and sequence extraction ===")


    os.makedirs(OUTPUT_DIR, exist_ok=True)


    pdb_files = glob.glob(f"{PDB_DIR}/*.pdb")
    if not pdb_files:
        print(f"No PDB files found in {PDB_DIR}.")
        return

    print(f"Found {len(pdb_files)} PDB files: {', '.join([os.path.basename(f) for f in pdb_files[:5]])}" +
          (f"..." if len(pdb_files) > 5 else ""))


    create_tcl_script()


    choice = input("Run VMD to separate chains? (y/n): ").lower()
    vmd_success = False

    if choice == 'y':
        vmd_success = run_vmd_script()
    else:
        print("Skipping VMD processing.")
        print(f"Ensure that separated structures exist under {OUTPUT_DIR}/protein_chains and {OUTPUT_DIR}/nucleic_chains.")


        if not os.path.exists(f"{OUTPUT_DIR}/protein_chains") or not os.path.exists(f"{OUTPUT_DIR}/nucleic_chains"):
            print("Required output directories are missing; separate the chains first.")
            return


    if vmd_success or choice != 'y':
        choice = input("Extract sequences? (y/n): ").lower()
        if choice == 'y':
            create_summary = input("Create combined FASTA files? (y/n): ").lower() == 'y'
            process_all_pdbs(create_summary)
        else:
            print("Skipping sequence extraction.")


if __name__ == "__main__":
    main()
