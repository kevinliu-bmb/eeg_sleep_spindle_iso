import os
import logging
import numpy as np
import mne
import xml.etree.ElementTree as ET


def load_eeg_data(base_path):
    """
    Load EEG data (EDF files), sleep stage labels, and artifact indicators from a structured directory.

    The function reads EEG recordings in EDF format along with their corresponding sleep stage and 
    artifact indicator files (stored as `.npz`) for two participant groups: Autism Spectrum Disorder (ASD)
    and Typically Developing (TD). Each group's data should reside within its own subfolder.

    Directory structure should follow this format explicitly:

    base_path/
    ├── edf/
    │   ├── ASD/
    │   └── TD/
    ├── sleep_stages/
    │   ├── ASD/
    │   └── TD/
    └── artifacts/
        ├── ASD/
        └── TD/

    Parameters
    ----------
    base_path : str
        Path to the root directory containing the 'edf', 'sleep_stages', and 'artifacts' subdirectories.

    Returns
    -------
    data : list of dict
        A list of dictionaries, each containing the following keys:
            - 'subject_id': str, subject identifier extracted from the EDF filename.
            - 'group': str, participant group ('ASD' or 'TD').
            - 'edf': mne.io.Raw, raw EEG data object loaded with MNE (not preloaded).
            - 'sleep_stages': np.ndarray, sleep stage labels loaded from the corresponding `.npz` file.
            - 'artifact_indicator': np.ndarray, binary artifact indicators loaded from the corresponding `.npz` file.
    """
    groups = ['ASD', 'TD']
    data = []

    for group in groups:
        edf_path = os.path.join(base_path, 'edf', group)
        stages_path = os.path.join(base_path, 'sleep_stages', group)
        artifact_path = os.path.join(base_path, 'artifacts', group)

        edf_files = sorted([f for f in os.listdir(edf_path) if f.lower().endswith('.edf')])

        for edf_file in edf_files:
            subject_id = os.path.splitext(edf_file)[0]

            stages_file = os.path.join(stages_path, f'sleep_stages_{subject_id}.npz')
            artifact_file = os.path.join(artifact_path, f'artifacts_{subject_id}.npz')
            edf_file_full = os.path.join(edf_path, edf_file)

            if not os.path.exists(stages_file):
                print(f"Missing sleep stage file for {subject_id}, skipping...")
                continue
            if not os.path.exists(artifact_file):
                print(f"Missing artifact file for {subject_id}, skipping...")
                continue

            edf_raw = mne.io.read_raw_edf(edf_file_full, preload=False, verbose=False)

            # Load the first available array from npz files automatically
            with np.load(stages_file) as stages_npz:
                sleep_stages = stages_npz[stages_npz.files[0]]

            with np.load(artifact_file) as artifacts_npz:
                artifact_indicator = artifacts_npz[artifacts_npz.files[0]]

            data.append({
                'subject_id': subject_id,
                'group': group,
                'edf': edf_raw,
                'sleep_stages': sleep_stages,
                'artifact_indicator': artifact_indicator
            })

            print(f"Loaded subject {subject_id} from group {group}")

    return data


def load_custom_montage(montage_path):
    """
    Load EEG channel montage from an XML file.

    Parameters
    ----------
    montage_path : str
        Path to the EEG montage XML file.

    Returns
    -------
    montage : mne.channels.DigMontage
        Custom montage object.
    """
    tree = ET.parse(montage_path)
    root = tree.getroot()
    namespace = {'ns': 'http://www.egi.com/coordinates_mff'}

    sensor_elements = root.findall('.//ns:sensor', namespaces=namespace)

    positions = []
    channel_names = []

    for sensor in sensor_elements:
        sensor_type = sensor.find('ns:type', namespaces=namespace).text
        if sensor_type == '0':
            number = int(sensor.find('ns:number', namespaces=namespace).text)
            x = float(sensor.find('ns:x', namespaces=namespace).text)
            y = float(sensor.find('ns:y', namespaces=namespace).text)
            z = float(sensor.find('ns:z', namespaces=namespace).text)

            positions.append([x, y, z])
            channel_names.append(f'EEG {number}')

    ch_pos = {name: pos for name, pos in zip(channel_names, positions)}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')

    return montage
