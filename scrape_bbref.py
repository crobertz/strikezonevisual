from collections import defaultdict
import re
from xmlrpc.client import Boolean
from bs4 import BeautifulSoup, element
from urllib import request
from typing import Tuple, List, Optional
from time import sleep
import json
from pathlib import Path
import pandas as pd
from datetime import date, timedelta
import strikezone

URL_PREFIX = 'https://www.baseball-reference.com'

# dict to map Baseball-Reference abbreviationso to MLB abbreviations
DICT_BBREF_TO_SAVANT = {
    'TOR': 'TOR', 
    'MIA': 'MIA', 
    'ARI': 'ARI', 
    'NYY': 'NYY', 
    'LAA': 'LAA', 
    'SDP': 'SD', 
    'PIT': 'PIT', 
    'BOS': 'BOS', 
    'KCR': 'KC', 
    'SFG': 'SF', 
    'PHI': 'PHI', 
    'LAD': 'LAD', 
    'CHW': 'CWS', 
    'CIN': 'CIN', 
    'ATL': 'ATL', 
    'CHC': 'CHC', 
    'COL': 'COL', 
    'WSN': 'WSH', 
    'DET': 'DET', 
    'STL': 'STL', 
    'HOU': 'HOU', 
    'OAK': 'OAK', 
    'TBR': 'TB', 
    'MIL': 'MIL', 
    'SEA': 'SEA', 
    'MIN': 'MIN', 
    'NYM': 'NYM', 
    'BAL': 'BAL', 
    'CLE': 'CLE', 
    'TEX': 'TEX'}

def get_games_list(year: str | int) -> element.ResultSet:
    """
    Get the raw list of MLB games for the given year.
    Each line can be further sanitized into a BbrefGame class.

    Args:
        year (str | int): Season to get list of games.

    Returns:
        element.ResultSet: List of games as an bs4.element.ResultSet object.
    """
    year = str(year)
    url = 'https://www.baseball-reference.com/leagues/majors/' + year + '-schedule.shtml'

    source = request.urlopen(url).read()
    soup = BeautifulSoup(source, features='lxml')

    return soup.find_all('p', class_='game')


def get_umpires(box_url_suffix:str) -> dict:
    """
    Returns a dict of positions and umpires for the game indicated by the box 
    score URL.
    Does not account for games where the umpires change position mid-game.

    Args:
        box_url_suffix (str): Suffix of box score URL. To be appended to 
        `URL_PREFIX` to form 'https://www.baseball-reference.com' + suffix

    Returns:
        dict: Dict of umpires with the key the position and value the corresponding umpire.
    """
    url = URL_PREFIX + box_url_suffix
    box_source = request.urlopen(url).read()
    box_soup = BeautifulSoup(box_source, features='lxml')

    umps_raw = re.search('Umpires.*', str(box_soup))

    try:
        umps_list = re.search('HP[^\.]*', umps_raw.group(0)).group(0).split(', ')

        umps_dict = {}
        for pos, ump in map(lambda x: x.split(' - '), umps_list):
            if ump == '(none)':
                ump = None
            umps_dict[pos] = ump
    except AttributeError:
        umps_dict = {}

    return umps_dict


def parse_date(box_url_suffix:str) -> Tuple[str, str]:
    """
    Given a box score URL suffix returns the date and game number.
    Box score URL suffix is of the form `\{home_abr}{YYYYDDMM}{game_num}.shtml`.
    Game number is 0 for normal and 1, 2 for doubleheaders.

    Args:
        box_url_suffix (str): Box score URL suffix of ending in `\{home_abr}{YYYYDDMM}{game_num}.shtml`

    Returns:
        Tuple[str, str]: Returns the date as `YYYY-MM-DD` and game number.
    """
    match_obj = re.search('(\d{4})(\d{2})(\d{2})(\d)(?=\.)', box_url_suffix)

    date = '-'.join(list(match_obj.group(i) for i in range(1,4)))
    game_num = match_obj.group(4)

    return date, game_num


def parse_team(team_tag:element.Tag) -> Tuple[str, str]:
    """
    Helper function for parsing the bs4.element.Tag object appearing in the 
    MLB schedules page containing team abbreviation and name.

    Args:
        team_tag (element.Tag): bs4.element.Tag object appearing in the MLB schedules 
        page containing team abbreviation and name.

    Returns:
        Tuple[str, str]: Pair of parsed abbreviation and team name.
    """
    team_search_pattern = '(?<=\/teams\/)\w+(?=\/)'
    abbr = re.search(team_search_pattern, team_tag.get('href')).group(0)
    name = team_tag.string

    return abbr, name


class BbrefGame():
    """
    Class for storing basic game information as scraped from Baseball-Reference.

    Attributes:
        date (str): Game date in `YYYY-MM-DD` format.
        game_num (str): Game number where `0` is regular and `1` and `2` are games 
        1 and 2 of a doubleheader respectively.
        home_abbr (str): Home team abbreviation.
        home_name (str): Home team name.
        away_abbr (str): Away team abbreviation.
        away_name (str): Away team name.
        box_url (str): URL for the box score page.
        umpires (dict): Dict for umpire data if available.
    """
    def __init__(self, home:element.Tag, away:element.Tag, box_url:element.Tag):
        """
        Constructor for `BbrefGame` object.
        Argument tags come from raw game lists as given by `get_games_list`.

        Args:
            home (element.Tag): bs4.element.Tag containing home team data.
            away (element.Tag): bs4.element.Tag containing away team data.
            box_url (element.Tag): bs4.element.Tag containing box score url data.
        """
        self.date, self.game_num = parse_date(str(box_url))

        team_search_pattern = '(?<=\/teams\/)\w+(?=\/)'
        self.home_abbr = re.search(team_search_pattern, home.get('href')).group(0)
        self.home_name = home.string
        self.away_abbr = re.search(team_search_pattern, away.get('href')).group(0)
        self.away_name = away.string

        self.box_url = box_url.get('href')
        self.umpires = {}
    
    def set_umpires(self):
        self.umpires = get_umpires(self.box_url)

    def __repr__(self) -> str:
        to_join = [f'{k}: {v}' for k, v in self.__dict__.items()]
        return '\n'.join(to_join)


def write_games_json(games_list:element.ResultSet, json_path:Path):
    """
    Given a raw `games_list` scraped from the MLB schedules page, parses 
    each game into a `BbrefGame` object which are then stored into a 
    `defaultdict` keyed by game date. The `defaultdict` are then written to 
    a JSON file.

    Args:
        games_list (element.ResultSet): Raw list of games as returned by 
        `get_games_list`.
        json_path (Path): Path to save JSON file.
    """
    games_dict = defaultdict(list)

    for game_data in games_list:
        test_game = game_data.find_all('a')
        date, _ = parse_date(test_game[-1].get('href'))

        if test_game[-1].string == 'Boxscore':
            game = BbrefGame(*test_game)
            games_dict[date].append(game.__dict__)
  
    with open(json_path, 'w') as json_out:
        json.dump(games_dict, json_out)


def generate_bbref_csv(list_bbrefgame: List[BbrefGame], df_out_dir: Path, out_name: str='out', sleep_period: int=1):
    """
    Scrapes Baseball-Reference for game data for the given date and writes to csv 
    with the column headers `game_date`, `home_team`, `away_team`, `game_number`, 
    and `umpire`.

    Args:
        list_bbrefgame (List[BbrefGame]): List of games to scrape data. Must support 
        `BbrefGame` fields `BbrefGame.date`, `BbrefGame.home_abbr`, `BbrefGame.away_abbr`, 
        and `BbrefGame.box_url`.
        df_out_dir (Path): Path to write output csv.
        out_name (str, optional): Name for output csv. Defaults to 'out'.
        sleep_period (int, optional): Seconds to sleep between requests. Defaults to 1.
    """
    total = len(list_bbrefgame)

    col_headers = ['game_date', 'home_team', 'away_team', 'game_number', 'umpire']

    df = pd.DataFrame(columns=col_headers)
    idx = 0
    df_out_path = Path(df_out_dir / (out_name + '.csv'))

    df_bad = pd.DataFrame(columns=col_headers)
    bad_idx = 0
    df_bad_out_path = Path(df_out_dir / 'bad')
    Path.mkdir(df_bad_out_path, exist_ok=True)

    for game in list_bbrefgame:
        game['umpires'] = get_umpires(game.get('box_url'))

        if game.get('umpires').get('HP', None) is None:
            df_bad.loc[bad_idx] =  [game.get('date'), game.get('home_abbr'), game.get('away_abbr'), game.get('game_num'), game.get('umpires').get('HP', None)]
            bad_idx += 1
            print(f"Bad game: {game.get('date')} {game.get('home_abbr')} @ {game.get('away_abbr')}")
        else:
            df.loc[idx] = [game.get('date'), game.get('home_abbr'), game.get('away_abbr'), game.get('game_num'), game.get('umpires').get('HP', None)]
            idx += 1
            
        print(f'Generated {idx} of {total}')
        print(f'Sleeping for {sleep_period} seconds...')
        sleep(sleep_period)

    if bad_idx > 0:
        bad_csv_path = df_bad_out_path / (out_name + '_bad' + '.csv')
        df_bad.to_csv(bad_csv_path, index=False)
        print(f'Wrote {bad_idx} bad games to {bad_csv_path}')

    if idx > 0:
        df.to_csv(df_out_path, index=False)
        print(f'Wrote {idx} of {total} games to {df_out_path}')


def scrape_bbref_to_csv(start_date:str, end_date:str, json_path: Optional[Path]=None, dict_games: dict={}, df_out_prefix: Path=Path(__file__).parent / 'bbref_games', sleep_period=1):
    """
    Scrapes Baseball-Reference by calling `generate_bbref_csv` from `start_date` 
    to `end_date` inclusive. Either loads a dict of games from `json_path` or uses 
    `dict_games` argument which must support `{keys: List[BbrefGame]` types.

    Args:
        start_date (str): Start date to scrape in ISO format string.
        end_date (str): End date to scrape in ISO format string.
        json_path (Optional[Path], optional): Path for JSON file to load as a dict 
        containing games to scrape. Defaults to None.
        dict_games (dict, optional): Dict containing games to scrape. Defaults to {}.
        df_out_prefix (Path, optional): Path for csv outputs. Defaults to 
        `Path(__file__).parent/'bbref_games'`
        sleep_period (int, optional): Seconds to sleep between requests. Defaults to 1.
    """
    start_date_obj = current_date_obj = date.fromisoformat(start_date)
    incr_day = timedelta(days=1)
    end_date_obj = date.fromisoformat(end_date)

    # load json from json_path as dict if given
    if json_path:
        with open(json_path) as games_json:
            dict_games = json.load(games_json)
    
    # df_out_prefix = Path(__file__).parent / 'bbref_games'
    Path.mkdir(df_out_prefix, exist_ok=True)

    while current_date_obj <= end_date_obj:
        current_date_iso = current_date_obj.isoformat()
        print(f'Generating games for {current_date_iso}')
        
        if games := dict_games.get(current_date_obj.isoformat(), None):
            generate_bbref_csv(games, df_out_prefix, out_name=current_date_iso)
        else:
            print(f'No games for {current_date_iso}... skipping')

        current_date_obj += incr_day

        if current_date_obj < end_date_obj:
            print(f'Sleeping for {sleep_period} seconds...')
            sleep(sleep_period)

    print(f'Finished generating from {start_date_obj.isoformat()} to {end_date_obj.isoformat()}')


def merge_csv(csv_dir: Path, fix_teamname=True, write: Boolean=True, out_path: Path=(Path(__file__).parent / 'bbref-merged.csv')) -> pd.DataFrame:
    """
    Takes csv files in `csv_dir` and returns a merged DataFrame.
    Assumes common headers.

    Args:
        csv_dir (Path): Path of directory containing csv files with common headers.
        write (Boolean, optional): If True will write merged DataFrame to csv. 
        Defaults to True.
        fix_teamname (bool, optional): Replace team abbreviations to be consistent 
        with MLB abbreviations. Defaults to True.
        out_path (Path, optional): Path for output csv file. 
        Defaults to (Path(__file__).parent / 'merged.csv').

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    to_merge = []

    for count, child in enumerate(csv_dir.glob('./*.csv')):
        print(f'Merging {child}...')
        to_merge.append(pd.read_csv(child))
    
    df_merged = pd.concat(to_merge, axis=0, ignore_index=True)
    print(f'Merged {count+1} files.')

    if fix_teamname:
        df_merged.replace(to_replace=DICT_BBREF_TO_SAVANT, inplace=True)
        print(f'Team abbreviations replaced with MLB abbreviations.')

    if write:
        df_merged.to_csv(out_path, index=False)
        print(f'Wrote merged csv to {out_path}')

    return df_merged


def main(start_date: date, end_date: date, json_path: Path, write: Boolean=True, csv_dir: Path=Path(__file__).parent / 'bbref_games', csv_out_name: str='bbref-merged.csv'):
    games_list = get_games_list(start_date.year)
    write_games_json(games_list, json_path)
    scrape_bbref_to_csv(start_date.isoformat(), end_date.isoformat(), json_path)
    csv_out_path = csv_dir / csv_out_name
    df_bbref = merge_csv(csv_dir, write=write, out_path=csv_out_path)
    


if __name__ == '__main__':
    START_DATE = '2022-04-07'
    END_DATE = '2022-04-08'

    JSON_NAME = 'date_to_games.json'
    JSON_PATH = Path(__file__).parent / JSON_NAME

    CSV_DIR = Path(__file__).parent / 'bbref_games'

    main(date.fromisoformat(START_DATE), date.fromisoformat(END_DATE), JSON_PATH, CSV_DIR)
