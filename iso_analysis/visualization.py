import logging
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import mne
import numpy as np
from iso_analysis.utils import bootstrap_ci


def plot_iso_single_subject_save(iso_df, subject_id, save_path):
    """
    Generate a single combined ISO plot for all channels and save the plot image.

    Parameters
    ----------
    iso_df : pd.DataFrame
        DataFrame containing ISO data for a single subject.

    subject_id : str
        Subject identifier used in the plot title.

    save_path : str
        Path (including filename) to save the generated plot.

    Returns
    -------
    None
    """
    freq = iso_df['frequency']
    channels = [col.split('_sigma')[0] for col in iso_df.columns if 'sigma_all' in col]

    plt.figure(figsize=(10, 6))

    # Plot each channel explicitly on the same plot
    for ch in channels:
        plt.plot(freq, iso_df[f'{ch}_sigma_all'], label=f'{ch}')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Relative Power')
    plt.title(f'Combined ISO Spectrum for Subject: {subject_id}')
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save explicitly
    plt.savefig(save_path, dpi=300)
    plt.close()

    logging.info(f"ISO plot saved to: {save_path}")


def plot_iso_topomap(edf_raw, spec_iso, freq_iso, subject_id, montage_path, save_path):
    """
    Generate and save topographical plots for ISO band power and peak frequency.

    Parameters
    ----------
    edf_raw : mne.io.Raw
        EEG data loaded using MNE.
    
    spec_iso : np.ndarray, shape (n_channels, 1, n_freqs)
        ISO spectrum computed by get_iso function.
    
    freq_iso : np.ndarray, shape (n_freqs,)
        ISO frequency bins.
    
    subject_id : str
        Identifier for the subject, used in plot titles and filenames.

    montage_path : str
        File path to the EEG channel montage (XML file).

    save_path : str
        File path to explicitly save the generated topomap figure.

    Returns
    -------
    None
    """
    # Load montage from XML
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

    edf_raw = edf_raw.copy().pick(picks=channel_names)
    edf_raw.set_montage(montage, on_missing='warn')

    # Compute ISO metrics
    iso_band = (freq_iso >= 0.005) & (freq_iso <= 0.03)
    iso_band_power = np.trapz(spec_iso[:, 0, iso_band], freq_iso[iso_band], axis=-1)
    iso_peak_freq = freq_iso[np.argmax(spec_iso[:, 0, :], axis=-1)]

    # Plotting without explicit scaling adjustments
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    mne.viz.plot_topomap(
        iso_band_power, edf_raw.info, axes=axes[0], cmap='viridis', show=False
    )
    axes[0].set_title('ISO Band Power (0.005–0.03 Hz)')

    mne.viz.plot_topomap(
        iso_peak_freq, edf_raw.info, axes=axes[1], cmap='plasma', show=False
    )
    axes[1].set_title('ISO Peak Frequency (Hz)')

    plt.suptitle(f'Subject: {subject_id}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Explicitly save the figure
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    logging.info(f"Topomap saved to: {save_path}")


def plot_group_iso_ci(
    iso_results_df, freq_iso, channel, group_col='group', ci=95, save_path=None
):
    """
    Generate and save ISO spectra plot with mean and CI for ASD and TD groups for a single channel.

    Parameters
    ----------
    iso_results_df : pd.DataFrame
        ISO results dataframe including subject and group data.
    freq_iso : np.ndarray
        ISO frequency bins.
    channel : str
        EEG channel name.
    group_col : str
        Column name indicating group labels.
    ci : int
        Confidence interval percentage.
    save_path : str or None
        File path to save the plot image.

    Returns
    -------
    None
    """
    channel_cols = [
        col for col in iso_results_df.columns if col.startswith(f'{channel}_sigma_all_')
    ]

    plt.figure(figsize=(10, 6))
    groups = iso_results_df[group_col].dropna().unique()

    for group in groups:
        group_data = iso_results_df[iso_results_df[group_col] == group][channel_cols].dropna()

        if group_data.empty or np.isnan(group_data.values).all():
            print(f"Warning: No valid data for group '{group}' in channel '{channel}'. Skipping.")
            continue

        mean_spectrum, lower_ci, upper_ci = bootstrap_ci(group_data.values, ci=ci)

        plt.plot(freq_iso, mean_spectrum, label=f'{group} Mean')
        plt.fill_between(freq_iso, lower_ci, upper_ci, alpha=0.3, label=f'{group} {ci}% CI')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Relative Power')
    plt.title(f'ISO Spectrum with {ci}% CI - {channel}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        logging.info(f"Group plot saved to {save_path}")

    plt.close()
