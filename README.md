# EEG Infraslow Oscillation (ISO) Analysis Pipeline

This project provides a structured pipeline for analyzing infraslow oscillations (ISO) in EEG recordings obtained during Non-REM (NREM) sleep. The analysis focuses on characterizing the ISO spectrum, computing relevant metrics (ISO band power and peak frequencies), and visualizing results with spectral and topographical plots. This pipeline was developed to analyze EEG data in Autism Spectrum Disorder (ASD) and Typically Developing (TD) participant groups.

## Project Structure

```{txt}
mgh_eeg_spindle/
├── iso_analysis/
│   ├── __init__.py
│   ├── analysis.py            # ISO analysis pipeline
│   ├── io.py                  # Data loading utilities
│   ├── visualization.py       # Visualization utilities
│   └── utils/
│       ├── __init__.py
│       ├── bootstrap.py       # Bootstrapping confidence intervals
│       └── logging.py         # Logging setup
├── data/                      # EEG data directory (populate with your data)
│   ├── edf/
│   │   ├── ASD/
│   │   └── TD/
│   ├── sleep_stages/
│   │   ├── ASD/
│   │   └── TD/
│   └── artifacts/
│       ├── ASD/
│       └── TD/
├── eeg_ch_coords/             # EEG electrode coordinates
│   ├── coordinates32.xml
│   └── coordinates64.xml
├── outputs/                   # Analysis results (auto-generated)
│   └── iso_results_<timestamp>/
│       ├── analysis.log
│       └── iso_results_<timestamp>/
│           ├── iso_analysis_all_subjects.csv
│           ├── iso_additional_metrics.csv
│           ├── spectra/
│           ├── group_plots/
│           └── subject_plots/
├── .gitignore
├── requirements.txt
├── main.py                    # Main executable script
└── README.md                  # Project documentation
```

## Installation

Clone the Repository

```{bash}
git clone https://github.com/kevinliu-bmb/mgh_eeg_spindle
cd mgh_eeg_spindle
```

## Install Dependencies

Create a virtual environment (recommended):

```{bash}
conda create --name eeg_iso python=3.10
conda activate eeg_iso
```

Install required packages:

```{bash}
pip install -r requirements.txt
```

## Data Structure

Organize your EEG data following the structure below:

```{txt}
data/
    ├── edf/
    │   ├── ASD/
    │   │   ├── subject1.edf
    │   │   └── subject2.edf
    │   └── TD/
    │       ├── subject3.edf
    │       └── subject4.edf
    ├── sleep_stages/
    │   ├── ASD/
    │   │   ├── sleep_stages_subject1.npz
    │   │   └── sleep_stages_subject2.npz
    │   └── TD/
    │       ├── sleep_stages_subject3.npz
    │       └── sleep_stages_subject4.npz
    └── artifacts/
        ├── ASD/
        │   ├── artifacts_subject1.npz
        │   └── artifacts_subject2.npz
        └── TD/
            ├── artifacts_subject3.npz
            └── artifacts_subject4.npz
```

## Usage

Run the Analysis Pipeline

Execute the analysis pipeline from the project root directory:

```{bash}
python main.py
```

The outputs will be saved in the outputs/ folder, organized by timestamped folders containing:

- Individual subject ISO spectral plots
- ISO band power and peak frequency topographical plots
- CSV files with computed ISO metrics
- Group-level ISO spectral plots with 95% confidence intervals (bootstrapped)

Main Parameters (editable in main.py):

- base_data_path: Path to the EEG data directory.
- output_base_path: Path where outputs are stored.
- Montage XML paths (coordinates32.xml and coordinates64.xml) depending on EEG channel number.

## Analysis Overview

The pipeline involves the following key steps:

 1. Data Loading: EEG recordings (EDF), sleep stages, and artifact indicators are loaded.
 2. Preprocessing: EEG data is segmented, sleep stage labels applied, and artifact epochs identified.
 3. ISO Analysis:
    1. Computation of ISO spectrum using Multitaper PSD estimation.
    2. Calculation of ISO band power (0.005–0.03 Hz) and ISO peak frequency.
 4. Visualization:
    1. Spectral plots for individual subjects.
    2. Topographical plots of ISO metrics (band power & peak frequency).
    3. Group-level ISO spectral plots with bootstrapped 95% confidence intervals.
 5. Results Export: ISO metrics and processed data are exported to CSV for further analysis.

## Code Organization

- iso_analysis/io.py: Functions for loading EEG data, sleep stages, artifacts, and EEG montage.
- iso_analysis/analysis.py: ISO computation logic (get_iso), bootstrapping confidence intervals.
- iso_analysis/plotting.py: Visualization functions for spectral and topographical plots.
- iso_analysis/utils.py: Utility functions supporting analysis and plotting.
- main.py: Entrypoint script for the analysis pipeline.

## Contributing

Please submit pull requests for new features, bug fixes, or improvements. Include clear commit messages and update documentation when necessary.

## Authors

- Kevin Liu (<kevin.liu2@childrens.harvard.edu>)
- Haoqi Sun (<hsun3@bidmc.harvard.edu>)

## License

Distributed under the MIT License. See LICENSE for more information.
