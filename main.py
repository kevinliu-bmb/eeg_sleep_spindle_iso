from iso_analysis.io import load_eeg_data
from iso_analysis.analysis import run_iso_analysis

def main():
    base_data_path = './data'
    montage_paths = {
        '32': 'eeg_ch_coords/coordinates32.xml',
        '64': 'eeg_ch_coords/coordinates64.xml'
    }
    output_base_path = './outputs'

    data_list = load_eeg_data(base_data_path)

    iso_results_df, additional_metrics_df = run_iso_analysis(
        data_list,
        montage_paths=montage_paths,
        output_base_path=output_base_path
    )

if __name__ == '__main__':
    main()