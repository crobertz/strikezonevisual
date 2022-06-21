import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from sklearn.neural_network import MLPClassifier


def df_description_query(df_base: pd.DataFrame, query_list: list) -> pd.DataFrame:
    """Takes a DataFrame and returns a boolean DataFrame with values indicating whether 
    the `description` value is contained in `query_list`.

    Use in combination with df_queried to produce a DataFrame which is queried on 
    `description`.

    Args:
        df_base (pd.DataFrame): DataFrame containing source StatCast data.
        query_list (list): List of allowed `description` values.

    Returns:
        pd.DataFrame: Boolean DataFrame indicating the query result for a row.
    """
    return df_base['description'].apply(lambda x: x in query_list)


def df_queried(df_base: pd.DataFrame, query_list: list, columns: list=None) -> pd.DataFrame:
    # """query df_base with specified columns and description terms in query_list"""
    """Takes a source DataFrame and returns a DataFrame queried on `description` with 
    values contained in `query_list`.

    Args:
        df_base (pd.DataFrame): DataFrame containing source StatCast data.
        query_list (list): List of allowed `description` values.
        columns (list, optional): Columns to include in return DataFrame. 
        Defaults to None which includes all columns.

    Returns:
        pd.DataFrame: DataFrame containing StatCast data where `description` is 
        contained in `query_list` and specified columns.
    """
    if not columns:
        columns = list(df_base.columns)
    return df_base[columns].loc[df_description_query(df_base, query_list)]


def create_sb_dfs(
    df_base: pd.DataFrame, 
    ball_queries: list = ['ball'],
    strike_queries: list = ['called_strike']
    ) -> tuple[pd.DataFrame]:
    """Takes a DataFrame and returns a DataFrame queried on ball type and a 
    DataFrame queried on strike types (default to called strikes). 

    Args:
        df_base (pd.DataFrame): DataFrame containing source StatCast data.
        ball_queries (list, optional): Ball types to query. Defaults to ['ball'].
        strike_queries (list, optional): Strike types to query. Defaults to ['called_strike'].

    Returns:
        tuple[pd.DataFrame]: Returns `df_balls`, `df_strikes` which are DataFrames containing 
        data queried on given ball types and strike types respectively.
    """
    columns = ['plate_x', 'plate_z']
    df_balls = df_queried(df_base, ball_queries, columns=columns).assign(count_as_strike=0)
    df_strikes = df_queried(df_base, strike_queries, columns=columns).assign(count_as_strike=1)
    
    return df_balls, df_strikes
    

def create_training_data(df_balls: pd.DataFrame, df_strikes: pd.DataFrame) -> tuple[np.array]:
    """Takes source DataFrame for balls and strikes and returns numpy arrays to be used 
    for training data.

    Args:
        df_balls (pd.DataFrame): DataFrame containing pitch data called balls.
        df_strikes (pd.DataFrame): DataFrame containing pitch data for strike types.

    Returns:
        tuple[np.array]: Returns training data `X_train`, `Y_train` as numpy arrays.
    """
    df_train = pd.concat([df_balls, df_strikes]).sample(frac=1)

    X_train = df_train[['plate_x', 'plate_z']].to_numpy()
    Y_train = df_train[['count_as_strike']].to_numpy()

    return X_train, Y_train


def train_szone(X_train: np.array, Y_train: np.array) -> MLPClassifier:
    """Trains a MLP classifier trained on given training data.

    Args:
        X_train (np.array): numpy array of X training data.
        Y_train (np.array): numpy array of Y training data.

    Returns:
        MLPClassifier: MLP classifier trained on given data.
    """
    clf = MLPClassifier()
    clf.fit(X_train, Y_train.ravel())
    return clf


def add_true_zone_avg(ax: plt.Axes, df_pitches: pd.DataFrame, extra_inch: bool=True) -> plt.Axes:
    """Takes an Axes object and overlays the true strike zone. The strike zone 
    width is defined to be the width of home plate which is 17 inches which is 
    then converted to feet. StatCast has `sz_top` and `sz_bot` fields for each 
    pitch, the average of which are taken over all pitches to generate a 
    "true" strike zone for visualization purposes. Optional `extra_inch` argument 
    indicates whether to add an extra inch to either side of the strike zone 
    which in reality is within the acceptable dimensions as opposed to the strict 
    zone width given in the definition.

    Args:
        ax (plt.Axes): Axes object to overlay strike zone.
        df_pitches (pd.DataFrame): Pitch data DataFrame to compute strike zone height.
        extra_inch (bool, optional): If `True` overlays another strike zone 
        with an extra inch on either side. Defaults to `True`.

    Returns:
        Axes: Axes object with strike zone overlaid.
    """

    # PLATE_X_PLATE_WIDTH = 0.8 # deprecated estimate of half plate width
    PLATE_X_PLATE_WIDTH = (17 / 12) / 2 # officially plate is 17 inches -> convert to feet
    sz_top_avg = df_pitches['sz_top'].mean()
    sz_bot_avg = df_pitches['sz_bot'].mean()

    ax.add_patch(patches.Rectangle((-PLATE_X_PLATE_WIDTH,sz_bot_avg), 
                                    2*PLATE_X_PLATE_WIDTH, sz_top_avg-sz_bot_avg, 
                                    edgecolor='r', facecolor='none'))
    
    if extra_inch:
        PLATE_X_PLATE_WIDTH_PLUS_INCH = (19 / 12) / 2 #extra inch on each edge
        ax.add_patch(patches.Rectangle((-PLATE_X_PLATE_WIDTH_PLUS_INCH,sz_bot_avg), 
                                    2*PLATE_X_PLATE_WIDTH_PLUS_INCH, sz_top_avg-sz_bot_avg, 
                                    edgecolor='k', facecolor='none'))

    return ax


def add_clf_zone(ax: plt.Axes, clf: MLPClassifier, X: np.array) -> plt.Axes:
    """Takes an Axes object and overlays the MLP trained strike zone by plotting 
    the decision boundary for the `MLPClassifier`.

    Args:
        ax (plt.Axes): Axes object to overlay the MLP strike zone.
        clf (MLPClassifier): `MLPClassifier` trained on balls and strikes data.
        X (np.array): numpy array of training data to generate mesh grid.

    Returns:
        plt.Axes: Axes object with the MLP strike zone overlaid.
    """
    
    # create mesh grid
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)
    return ax


def sb_scatter(ax: plt.Axes, df_balls: pd.DataFrame, df_strikes: pd.DataFrame, 
                ball_queries: list=['ball'], strike_queries: list=['called_strike']) -> plt.Axes:
    """Takes an Axes object and overlays the scatter plot of balls and strikes.
    Ball and strike types are given as `ball_queries` and `strike_queries`.

    Args:
        ax (plt.Axes): Axes object to overlay scatter plot.
        df_balls (pd.DataFrame): Pitch DataFrame of balls.
        df_strikes (pd.DataFrame): Pitch DataFrame of strikes.
        ball_queries (list, optional): Ball types to query. Defaults to `['ball']`.
        strike_queries (list, optional): Strike types to query. Defaults to 
        `['called_strike']`.

    Returns:
        plt.Axes: Axes object with the scatter plot overlaid.
    """
    balls = ax.scatter(df_balls['plate_x'], df_balls['plate_z'], c='b', alpha=1)
    strikes = ax.scatter(df_strikes['plate_x'], df_strikes['plate_z'], c='y', alpha=1)    
    ax.legend((balls, strikes), ('/'.join(ball_queries), '/'.join(strike_queries)),
                loc='lower left')
    return ax


def generate_plot(clf: MLPClassifier, df_base: pd.DataFrame, df_balls: pd.DataFrame, 
                    df_strikes: pd.DataFrame, X_train: np.array,
                    ball_queries: list=['ball'], strike_queries: list=['called_strike'],
                    figsize: tuple=(10,10), title: str='', extra_inch: bool=True) -> plt.Axes:
    """Generates a scatter plot with balls and strikes, true strike zone 
    overlaid, and the decision boundary of the `MLPClassifier` trained 
    on the balls and strikes data.

    Args:
        clf (MLPClassifier): `MLPClassifier` trained on balls and strikes data.
        df_base (pd.DataFrame): Pitch DataFrame to be used to compute `sz_top` and 
        `sz_bot` averages.
        df_balls (pd.DataFrame): Pitch DataFrame queried for balls.
        df_strikes (pd.DataFrame): Pitch DataFrame queried for strikes.
        X_train (np.array): numpy array of training data used to generate 
        decision boundary.
        ball_queries (list, optional): Ball types to query. Defaults to ['ball'].
        strike_queries (list, optional): Strike types to query. Defaults to 
        ['called_strike'].
        figsize (tuple, optional): Figure size. Defaults to (10,10).
        title (str, optional): Title of plot. Defaults to ''.
        add_extra_inch (bool, optional): Include extra inch on both sides of plate for 
        strikezone overlay. Defaults to True.

    Returns:
        plt.Axes: Axes object with generated plots.
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax = sb_scatter(ax, df_balls, df_strikes, ball_queries, strike_queries)
    ax = add_true_zone_avg(ax, df_base, extra_inch=extra_inch)
    ax = add_clf_zone(ax, clf, X_train)
    ax.set_title(title)

    ax.set_xlim([-4,4])
    ax.set_ylim([-1,6])

    return ax


# TODO: can generalize to not just umps and take general base bf and query terms
def generate_ump_zones(csv_name: Path, ump_name: Optional[str] = None, extra_inch: bool=False,
    out_dir: Optional[Path] = None) -> None:
    """
    Generates strikezone visuals for umpires and saves to file.

    Args:
        csv_name (Path): Path for StatCast csv with umpire data merged.
        ump_name (Optional[str], optional): Name of umpire to generate visual. 
        If None then generates for all umpires. Defaults to None.
        extra_inch (bool, optional): _description_. If True generates true strikezone 
        with an extra inch on each side of the plate. Defaults to False.
        out_dir (Optional[Path], optional): Output directory. If None outputs to 
        `out` subdirectory created in the current directory. Defaults to None.
    """

    csv_path = Path(__file__).parent / csv_name
    df_base = pd.read_csv(csv_path).dropna(subset=['plate_x', 'plate_z'])
        
    if ump_name is None:
        umps = df_base['umpire'].unique()
    else:
        umps = [ump_name]

    total = len(umps)

    if out_dir is None:
        out_dir = Path(__file__).parent / 'out'
        out_dir.mkdir(exist_ok=True)

    for i, ump in enumerate(umps):
        print(f'Generating zone for {ump}: {i+1} of {total}')

        # set up required DataFrames
        df_ump = df_base[df_base['umpire'] == ump]
        df_balls, df_strikes = create_sb_dfs(df_ump)
        X_train, Y_train = create_training_data(df_balls, df_strikes)
        
        # train clf zone and generate plots
        clf = train_szone(X_train, Y_train)
        ax = generate_plot(clf, df_ump, df_balls, df_strikes, X_train, title=ump, extra_inch=extra_inch)
        
        # save output figure
        plt.savefig(out_dir / re.sub(r'\W+', '_', ump))

        print(f'Generated zone for {ump}: {i+1} of {total}')
    
    print(f'Generated all {total} zones.')


def generate_ev_paths(ev_dir: Path) -> List[Path]:
    """
    Takes Retrosheet event file directory Path and returns list of
    `Path`s for event files.

    Returns:
        List[Path]: List of `Path`s for each individual event files.
    """

    ev_files = []
    
    for child in ev_dir.iterdir():
        if child.suffix in ('.EVA', '.EVN'):
            ev_files.append(child)
    
    return ev_files


def create_event_df(ev_files: List[Path], fix_teamname=True) -> pd.DataFrame:
    """    
    Take a list of `Path`s to Retrosheet event files and generates a DataFrame 
    with columns `date`, `home`, `visit`, `number`, `ump`.

    Columns:
        `date`: Game date
        `home`: Home team
        `visit`: Visiting team
        `number`: Game number. `0` indicates non-doubleheader while `1` and 
        `2` indicates games 1 and 2 of a doubleheader respectively.
        `ump`: Home plate umpire for the game.

    Args:
        ev_files (List[Path]): List of `Path`s to the event files.
        fix_teamname (bool, optional): Replace Retrosheet team abbreviations 
        with MLB abbreviations. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing game data with columns `date`, `home`, 
        `visit`, `number`, and `ump`.
    """
    col_headers = ['game_date', 'home_team', 'away_team', 'game_number', 'umpire']
    df = pd.DataFrame(columns=col_headers)
    idx = 0

    RETRO_TEAM_TO_SAVANT = {
        'SDN': 'SD',
        'CHN': 'CHC',
        'SFN': 'SF',
        'NYA': 'NYY',
        'KCA': 'KC',
        'LAN': 'LAD',
        'SLN': 'STL',
        'ANA': 'LAA',
        'TBA': 'TB',
        'NYN': 'NYM',
        'WAS': 'WSH',
        'CHA': 'CWS'}

    for child in ev_files:
        pbp_path = child
        with open(pbp_path, 'r') as pbp_file:
            for line in pbp_file:
                line = line.split(',')
                data = line[1]
                match data:
                    # ordering-dependent on event file format
                    case 'visteam':
                        visit = line[2].strip()
                        if fix_teamname and visit in RETRO_TEAM_TO_SAVANT:
                            visit = RETRO_TEAM_TO_SAVANT[visit]
                    case 'hometeam':
                        home = line[2].strip()
                        if fix_teamname and home in RETRO_TEAM_TO_SAVANT:
                            home = RETRO_TEAM_TO_SAVANT[home]
                    case 'date':
                        date = line[2].strip()
                        date = re.sub(r'/', '-', date) #convert to YYYY-MM-DD
                    case 'number':
                        number = int(line[2].strip())
                    case 'umphome':
                        ump = line[2].strip()
                        df.loc[idx] = [date, home, visit, number, ump]
                        idx += 1

    return df             


def umpid_to_name(event_df: pd.DataFrame, ump_json_path: Path) -> None:
    """
    Given event DataFrame generated by `create_event_df` and `Path` to a JSON 
    file mapping Retrosheet umpire ID to names, replaces umpire ID in event 
    DataFrame to corresponding names.

    Args:
        event_df (pd.DataFrame): Retrosheet event DataFrame generated by 
        `create_event_df`
        ump_json_path (Path): `Path` to JSON file mapping Retrosheet umpire ID to 
        names
    """

    with open(ump_json_path) as ump_json:
        ump_id_to_ump_dict = json.load(ump_json)

    event_df.replace(to_replace=ump_id_to_ump_dict, inplace=True)


def generate_df_games(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a DataFrame containing only `game_date`, `home_team`, `away_team`, 
    and `game_pk` information.

    Args:
        df_base (pd.DataFrame): Source StatCast DataFrame

    Returns:
        pd.DataFrame: DataFrame containing only `game_date`, `home_team`, `away_team`, 
        and `game_pk`.
    """
    return df_base[['game_date', 'home_team', 'away_team', 'game_pk']].drop_duplicates()   


def generate_dict_dh_game_pk(df_games: pd.DataFrame) -> defaultdict[list]:
    """
    Given a DataFrame containing game data returns a dict containing data for 
    doubleheaders.

    Args:
        df_games (pd.DataFrame): StatCast DataFrame containing game data

    Returns:
        defaultdict[list]: dict containing doubleheader data
            key: (`game_date`, `home_team`, `away_team`)
            value: list of `game_pk` for doubleheader with the game data as given 
            by key
    """
    # can probably be optimized
    game_dict = defaultdict(list)

    for row in df_games.iterrows():
        _, data = row
        game = (data['game_date'], data['home_team'], data['away_team'])
        game_dict[game].append(data['game_pk'])

    return {k:v for k, v in game_dict.items() if len(v) > 1}


def generate_dict_game_pk_to_game_number(dict_dh: defaultdict[list]) -> dict:
    """
    Takes a dict containing doubleheader data as given by `generate_dict_dh_game_pk` 
    and returns a dict with keys `game_pk` and corresponding game number. Game 
    number is `1` for game 1 and `2` for game 2 and assigned based on the assumption 
    that the earlier game has a lower `game_pk`.

    Args:
        dict_dh (defaultdict[list]): dict of doubleheader data as given by
        `generate_dict_dh_game_pk`.

    Returns:
        dict: 
            key: `game_pk`
            value: Game number for the corresponding game. `1` indicates game 1 and 
            `2` indicates game 2.
    """
    # can probably be optimized

    game_pk_to_game_number = {}
    
    for v in dict_dh.values():
        for i, game_pk in enumerate(sorted(v)):
            game_pk_to_game_number[game_pk] = i + 1
    
    return game_pk_to_game_number


# helper to map game_pk to game_number, probably unnecessary
def game_pk_to_game_number(game_pk: int, dict_game_pk_to_game_number: dict) -> int:
    """
    Helper function that takes a `game_pk` and a dict mapping `game_pk` to game
    number for doubleheaders and returns the corresponding game number.

    Args:
        game_pk (int): `game_pk` for the game
        dict_game_pk_to_game_number (dict): dict mapping `game_pk` to game number 
        for doubleheaders.

    Returns:
        int: Game number for the game with given `game_pk`.
        It is `0` for non-doubleheader, `1` for doubleheader game 1, and `2` for 
        doubleheader game 2.
    """
    if game_pk in dict_game_pk_to_game_number:
        return dict_game_pk_to_game_number[game_pk]
    return 0


def merge_df_base_with_df_retro(df_base: pd.DataFrame, df_retro: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a source StatCast DataFrame `df_base` and a Retrosheet DataFrame containing 
    columns `game_date`, `home_team`, `away_team`, `game_number`. Effectively adds 
    columns for doubleheader data and umpires in the source DataFrame.

    Will not account for games with multiple home plate umpires.

    Args:
        df_base (pd.DataFrame): Source StatCast Dataframe
        df_retro (pd.DataFrame): Retrosheet DataFrame as given by `create_event_df`

    Returns:
        pd.DataFrame: DataFrame with source StatCast data with doubleheader and umpire 
        data added
    """
    # may be possible to optimize

    df_games = generate_df_games(df_base)
    dict_dh = generate_dict_dh_game_pk(df_games)
    dict_game_pk_to_game_number = generate_dict_game_pk_to_game_number(dict_dh)
    
    # create game_number column
    df_base['game_number'] = df_base['game_pk'].apply(
        lambda x: game_pk_to_game_number(x, dict_game_pk_to_game_number))

    merge = df_base.drop(columns='umpire').merge(
        df_retro, 
        on=['game_date', 'home_team', 'away_team', 'game_number'])
        
    return merge


def generate_statcast_df_with_umpires(
    ev_dir: Path, 
    statcast_csv_path: Path, 
    ump_json_path: Path, 
    save_to_csv: bool = True,
    save_path: Path = Path.cwd() / 'merged.csv') -> pd.DataFrame:
    """
    Takes `Path`s to Retrosheet event files directory, StatCast CSV file, and the JSON file 
    mapping Retrosheet umpire ID's to umpire names and returns the StatCast CSV file with 
    an additional `umpire` column with the plate umpire for the game. Games with multiple 
    home plate umpires are dropped.

    Args:
        ev_dir (Path): `Path` to the directory with Retrosheet event files
        statcast_csv_path (Path): `Path` to the StatCast CSV file
        ump_json_path (Path): `Path` to the umpire ID to name JSON file
        save_to_csv (bool, optional): Saves the output DataFrame to CSV. Defaults to True.
        save_path (Path, optional): Path for saving the csv. Defaults to Path.cwd().

    Returns:
        pd.DataFrame: _description_
    """
    ev_paths = generate_ev_paths(ev_dir)
    ev_df = create_event_df(ev_paths)
    umpid_to_name(ev_df, ump_json_path)
    df_base = pd.read_csv(statcast_csv_path)
    df_merged = merge_df_base_with_df_retro(df_base, ev_df)

    if save_to_csv:
        df_merged.to_csv(save_path)

    return df_merged