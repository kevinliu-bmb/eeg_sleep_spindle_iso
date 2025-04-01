import os
from datetime import datetime
import logging
from iso_analysis.io import load_eeg_data
from iso_analysis.analysis import run_iso_analysis
from iso_analysis.utils.logging import setup_logging


def main():
    base_data_path = './data'
    montage_paths = {
        '32': 'eeg_ch_coords/coordinates32.xml',
        '64': 'eeg_ch_coords/coordinates64.xml'
    }

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join('./outputs', f'iso_results_{timestamp}')
    os.makedirs(output_folder, exist_ok=True)

    # Initialize logging explicitly within the output folder
    setup_logging(output_folder)
    logging.info('Starting ISO analysis pipeline.')

    # Load EEG data
    data_list = load_eeg_data(base_data_path)

    # Run analysis pipeline
    iso_results_df, additional_metrics_df = run_iso_analysis(
        data_list,
        montage_paths=montage_paths,
        output_base_path=output_folder
    )

    logging.info('ISO analysis pipeline completed successfully.')


if __name__ == '__main__':
    main()
