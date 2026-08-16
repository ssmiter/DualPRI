import os
import numpy as np
import pickle
from tqdm import tqdm

def parse_pssm_to_array(pssm_file):
    'Parse pssm to array.'
    with open(pssm_file, 'r') as file:
        lines = file.readlines()

    pssm_data = []
    matrix_started = False

    for line in lines:

        if line.strip() == "":
            continue


        if "Last position-specific scoring matrix" in line:
            matrix_started = True
            continue


        if matrix_started:
            matrix_started = False
            continue


        parts = line.strip().split()


        if len(parts) < 22:
            continue

        try:

            scores = []
            for value in parts[2:22]:

                if not value.replace('-', '').isdigit():
                    continue
                scores.append(int(value))

            if len(scores) == 20:
                pssm_data.append(scores)

        except (ValueError, IndexError):
            continue


    if not pssm_data:
        for line in lines:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                try:
                    scores = [int(x) for x in parts[:20]]
                    if len(scores) == 20:
                        pssm_data.append(scores)
                except (ValueError, IndexError):
                    continue

    if not pssm_data:
        raise ValueError(f"Unable to parse PSSM data from file: {pssm_file}")

    return np.array(pssm_data)


def process_pssm_files(directory):
    'Process pssm files.'
    pssm_arrays = {}
    errors = []

    total = len(directory)
    with tqdm(total=total, desc="Processing pssm files", ncols=100) as pbar:
        for filename in os.listdir(directory):
            if filename.endswith(".pssm"):
                file_path = os.path.join(directory, filename)
                try:
                    pssm_array = parse_pssm_to_array(file_path)
                    pssm_arrays[filename] = pssm_array


                    pbar.set_postfix({
                        'Current': f"{filename}"
                    })
                except Exception as e:
                    errors.append(f"Error processing {filename}: {str(e)}")
                    continue
                finally:
                    pbar.update(1)


    with open(os.path.join(directory, f'pssm_s{dataset}.pkl'), 'wb') as pickle_file:
        pickle.dump(pssm_arrays, pickle_file)

    print(f"Successfully processed {len(pssm_arrays)} PSSM files")
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(error)

dataset = '394'
# dataset = 'myoglobin'

# Replace with your directory containing .pssm files
pssm_directory = f'./Dataset/PSSM_{dataset}'
process_pssm_files(pssm_directory)
