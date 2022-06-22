from pathlib import Path

import pandas as pd

import strikezone

if __name__ == '__main__':
    """
    Example script for visualizing.
    Assumes there is a csv file `./merged.csv` containing StatCast data 
    along with umpire data.
    """

    # load df from merged csv at specified path
    CSV_NAME = 'statcast_with_umpires.csv'
    CSV_PATH = Path(__file__).parent / CSV_NAME
    df_base = pd.read_csv(CSV_PATH)

    # ump to generate, None to generate for all umps
    UMP = 'Angel Hernandez'

    strikezone.generate_ump_zones(CSV_NAME, ump_name=UMP)