import logging
import os
import sys

def setup_logging(output_folder):
    """
    Set up logging to output logs to both a file and the terminal.

    Parameters
    ----------
    output_folder : str
        Path where log file will be stored.
    """
    os.makedirs(output_folder, exist_ok=True)
    log_file = os.path.join(output_folder, 'analysis.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
