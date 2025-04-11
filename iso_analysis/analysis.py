import os
from datetime import datetime
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import groupby
from scipy.signal import detrend
from scipy.stats import mode
from scipy.interpolate import interp1d
from mne.time_frequency import psd_array_multitaper
from iso_analysis.io import load_custom_montage
from iso_analysis.visualization import (
    plot_iso_single_subject_save,
    plot_iso_topomap,
    plot_group_iso_ci
)


def get_iso(edf, sleep_stages, artifact_indicator, ch_groups):
    """
    Computes the infraslow oscillation (ISO) spectrum from EEG data during Non-REM (NREM) sleep epochs.

    Parameters
    ----------
    edf : mne.io.Raw
        The EEG data object loaded using MNE, containing EEG signals and sampling frequency.

    sleep_stages : np.ndarray, shape (n_samples,)
        Numeric array indicating sleep stages at each EEG sample. Expected coding:
        0 = NREM, 1 = REM, 2 = Wake, 3 = Artifact.
        (Sleep stages will be resampled internally to match EEG data length.)

    artifact_indicator : np.ndarray, shape (n_samples,)
        Binary array (1-dimensional) indicating artifacts at each EEG sample (1 = artifact, 0 = clean).
        (Artifacts are resampled internally to match EEG data length.)

    ch_groups : list of list of str
        List defining channel groupings for analysis. Each inner list contains channel names to average.

    Returns
    -------
    spec_iso : np.ndarray, shape (n_channels, 3, n_freqs)
        ISO spectrum, separately computed for sigma_all, sigma_slow, and sigma_fast bands, for each channel group.

    freq_iso : np.ndarray, shape (n_freqs,)
        Frequency bins corresponding to ISO power spectra, typically in the infraslow range (0–0.1 Hz).
    """
    eeg = edf.get_data() * 1e6
    original_fs = edf.info["sfreq"]
    fs = original_fs

    # Segment EEG into epochs
    window_size = int(round(4 * fs))
    step_size = int(round(2 * fs))
    epoch_time = 2
    start_ids = np.arange(0, eeg.shape[1] - window_size + 1, step_size)
    epochs = np.array([eeg[:, x : x + window_size] for x in start_ids])

    # Assign sleep stages and artifact indicators per epoch (identical logic to original function)
    sleep_stages = np.array(
        [
            mode(sleep_stages[x : x + window_size], keepdims=False).mode
            for x in start_ids
        ]
    )
    artifact_indicator = np.array(
        [artifact_indicator[x : x + window_size].any(axis=0) for x in start_ids]
    )

    sleep_ids = np.where(sleep_stages == 0)[0]
    if len(sleep_ids) == 0:
        raise ValueError("No NREM epochs found.")
    start = sleep_ids[0]
    end = sleep_ids[-1] + 1
    epochs = epochs[start:end]
    sleep_stages = sleep_stages[start:end]
    artifact_indicator = artifact_indicator[start:end]

    # Detrend epochs
    epochs = detrend(epochs, axis=-1)

    # Calculate sigma-band power
    spec, freq = psd_array_multitaper(
        epochs,
        sfreq=fs,
        fmin=11,
        fmax=15,
        bandwidth=1,
        normalization="full",
        remove_dc=True,
        verbose=False,
    )
    spec[np.isinf(spec)] = np.nan
    dfreq = freq[1] - freq[0]

    sigma_db = 10 * np.log10(np.sum(spec, axis=-1) * dfreq).T
    sigma_db[np.isinf(sigma_db)] = np.nan
    fs_sigma = 1 / epoch_time

    sigma_db = np.array(
        [
            np.nanmean(sigma_db[[edf.ch_names.index(x) for x in xx]], axis=0)
            for xx in ch_groups
        ]
    )

    # Artifact_indicator is 1-dimensional; original assumed 2D channel-specific artifacts
    artifact_indicator2 = np.array([artifact_indicator for _ in ch_groups])

    window_size = int(round(256 * fs_sigma))
    step_size = int(round(64 * fs_sigma))
    freq_iso = np.linspace(0, 0.1, 101)[1:]
    spec_iso_all_ch = []

    for chi in range(len(ch_groups)):
        good_ids = (sleep_stages == 0) & (~artifact_indicator2[chi])
        spec_isos = []
        cc = 0
        for k, l in groupby(good_ids):
            ll = len(list(l))
            if not k:
                cc += ll
                continue
            for start in np.arange(cc, cc + ll - window_size + 1, step_size):
                xx = np.array(
                    [
                        sigma_db[chi, start : start + window_size],
                    ]
                )

                # Skip empty or all-NaN epochs explicitly
                if np.isnan(xx).all() or np.all(xx == 0):
                    continue

                xx = detrend(xx, axis=-1)
                spec_iso, freq_out = psd_array_multitaper(
                    xx,
                    fs_sigma,
                    fmin=0,
                    fmax=0.2,
                    bandwidth=0.01,
                    normalization="full",
                    verbose=False,
                )

                ff = interp1d(
                    freq_out,
                    spec_iso,
                    axis=-1,
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                spec_iso = ff(freq_iso)

                spec_iso /= spec_iso.sum(axis=-1, keepdims=True)
                spec_isos.append(spec_iso)
            cc += ll

        # Explicit handling if no valid epochs found
        if len(spec_isos) == 0:
            logging.warning(f"No valid epochs for channel {ch_groups[chi]}. Returning NaN.")
            spec_iso_ch = np.full((1, len(freq_iso)), np.nan)
        else:
            spec_iso_ch = np.nanmean(np.array(spec_isos), axis=0)

        spec_iso_all_ch.append(spec_iso_ch)

    spec_iso = np.array(spec_iso_all_ch)
    
    return spec_iso, freq_iso


def run_iso_analysis(
    data_list,
    montage_paths={
        "32": "eeg_ch_coords/coordinates32.xml",
        "64": "eeg_ch_coords/coordinates64.xml",
    },
    output_base_path="./outputs",
):
    """
    Full pipeline to run ISO analysis on all subjects, saving all plots and metrics explicitly.

    Parameters
    ----------
    data_list : list of dict
        EEG data loaded with load_eeg_data function.

    montage_paths : dict
        Dictionary with keys '32' and '64' pointing to corresponding montage XML files.

    output_base_path : str
        Directory path to save results explicitly.

    Returns
    -------
    iso_results_df : pd.DataFrame
        ISO spectral results DataFrame.

    additional_metrics_df : pd.DataFrame
        ISO band power and peak frequency DataFrame.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output_base_path, f"iso_results_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    # Create organized subdirectories explicitly
    spectra_dir = os.path.join(output_folder, "spectra")
    topomaps_dir = os.path.join(output_folder, "topomaps")
    group_plot_dir = os.path.join(output_folder, "group_plots")

    os.makedirs(spectra_dir, exist_ok=True)
    os.makedirs(topomaps_dir, exist_ok=True)
    os.makedirs(group_plot_dir, exist_ok=True)

    logging.info(f"Results will be saved to: {output_folder}")

    iso_results = []
    additional_metrics = []
    freq_iso = None

    for idx, subject in enumerate(tqdm(data_list, desc="Running ISO Analysis")):
        subject_id = subject["subject_id"]
        group = subject["group"]

        tqdm.write(f"[{idx+1}/{len(data_list)}] Processing subject: {subject_id} ({group})")
        logging.info(f"Processing subject: {subject_id} ({group})")

        try:
            edf_raw = subject["edf"].copy().load_data()
            sleep_stages = subject["sleep_stages"]
            artifact_indicator = subject["artifact_indicator"]

            # Choose correct montage explicitly
            n_channels = len(edf_raw.ch_names) - ("EEG VREF" in edf_raw.ch_names)
            montage_key = "64" if n_channels > 32 else "32"
            montage_path = montage_paths[montage_key]

            montage = load_custom_montage(montage_path)
            edf_raw.pick(picks=[ch for ch in edf_raw.ch_names if ch != "EEG VREF"])
            edf_raw.set_montage(montage, on_missing="warn")

            ch_groups = [[ch] for ch in edf_raw.ch_names]

            # Run ISO analysis explicitly
            spec_iso, freq_iso = get_iso(
                edf_raw, sleep_stages, artifact_indicator, ch_groups
            )

            # Save ISO spectrum data explicitly
            iso_dict = {"subject_id": subject_id, "group": group}
            for ch_idx, ch_name in enumerate(edf_raw.ch_names):
                for f_idx, freq in enumerate(freq_iso):
                    col_label = f"{ch_name}_sigma_all_{freq:.4f}Hz"
                    iso_dict[col_label] = spec_iso[ch_idx, 0, f_idx]
            iso_results.append(iso_dict)

            # ISO spectrum plot explicitly saved in spectra_dir
            iso_df_single_subject = pd.DataFrame({"frequency": freq_iso})
            for ch_idx, ch_name in enumerate(edf_raw.ch_names):
                iso_df_single_subject[f"{ch_name}_sigma_all"] = spec_iso[ch_idx, 0, :]

            spectrum_plot_path = os.path.join(
                spectra_dir, f"{subject_id}_iso_spectrum.png"
            )
            plot_iso_single_subject_save(
                iso_df_single_subject, subject_id, spectrum_plot_path
            )

            # ISO topomap explicitly saved in topomaps_dir
            topomap_plot_path = os.path.join(
                topomaps_dir, f"{subject_id}_iso_topomaps.png"
            )
            plot_iso_topomap(
                edf_raw, spec_iso, freq_iso, subject_id, montage_path, topomap_plot_path
            )

            # Additional metrics explicitly computed and saved
            band_power_dict = {"subject_id": subject_id, "group": group}
            peak_freq_dict = {"subject_id": subject_id, "group": group}
            iso_band = (freq_iso >= 0.005) & (freq_iso <= 0.03)

            for ch_idx, ch_name in enumerate(edf_raw.ch_names):
                iso_power = spec_iso[ch_idx, 0, :]
                auc_band_power = np.trapz(iso_power[iso_band], freq_iso[iso_band])
                peak_freq = freq_iso[np.argmax(iso_power)]

                band_power_dict[f"{ch_name}_ISO_bandpower_0.005-0.03Hz"] = auc_band_power
                peak_freq_dict[f"{ch_name}_ISO_peak_frequency"] = peak_freq

            # Relative ISO Band Power Calculation
            relative_iso_band_dict = {"subject_id": subject_id, "group": group}
            iso_total_band = (freq_iso >= 0.001) & (freq_iso <= 0.1)
            iso_specific_band = (freq_iso >= 0.005) & (freq_iso <= 0.03)

            for ch_idx, ch_name in enumerate(edf_raw.ch_names):
                iso_power = spec_iso[ch_idx, 0, :]
                total_auc_power = np.trapz(iso_power[iso_total_band], freq_iso[iso_total_band])
                specific_auc_power = np.trapz(iso_power[iso_specific_band], freq_iso[iso_specific_band])
                relative_iso_power = (specific_auc_power / total_auc_power) * 100 if total_auc_power != 0 else np.nan
                relative_iso_band_dict[f"{ch_name}_relative_ISO_power_0.005-0.03Hz"] = relative_iso_power

            additional_metrics.append({**band_power_dict, **peak_freq_dict, **relative_iso_band_dict})

        except Exception as e:
            tqdm.write(f"Error processing subject {subject_id}: {e}")
            logging.error(f"Error processing subject {subject_id}: {e}", exc_info=True)

    # Save ISO results explicitly
    iso_results_df = pd.DataFrame(iso_results)
    iso_results_csv = os.path.join(output_folder, "iso_analysis_all_subjects.csv")
    iso_results_df.to_csv(iso_results_csv, index=False)
    logging.info(f"\nISO spectra results saved: {iso_results_csv}")

    # Save additional metrics explicitly
    additional_metrics_df = pd.DataFrame(additional_metrics)
    metrics_csv = os.path.join(output_folder, "iso_additional_metrics.csv")
    additional_metrics_df.to_csv(metrics_csv, index=False)
    logging.info(f"Additional ISO metrics saved: {metrics_csv}")

    # Generate and explicitly save group-wise ISO plots
    for channel in edf_raw.ch_names:
        group_plot_path = os.path.join(group_plot_dir, f"ISO_CI_{channel}.png")
        plot_group_iso_ci(iso_results_df, freq_iso, channel, save_path=group_plot_path)

    return iso_results_df, additional_metrics_df
