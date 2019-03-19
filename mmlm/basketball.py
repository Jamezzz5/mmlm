import itertools
import numpy as np
import pandas as pd
import mmlm.model as md


class Team(object):
    def __init__(self, team_dict):
        for k in team_dict:
            setattr(self, k, team_dict[k])
        self.games = {}

class Teams(object):
    Rk = 'Rk'
    Team = 'Team'
    Conf = 'Conf'
    W_L = 'W-L'
    AdjEM = 'AdjEM'
    AdjO = 'AdjO'
    AdjO_Rank = 'AdjO- Rank'
    AdjD = 'AdjD'
    AdjD_Rank = 'AdjD- Rank'
    AdjT = 'AdjT'
    AdjT_Rank = 'AdjT- Rank'
    Luck = 'Luck'
    Luck_Rank = 'Luck- Rank'
    AdjEM_SOS = 'AdjEM_SOS'
    AdjEM_Rank = 'AdjEM_SOS- Rank'
    OppO_SOS = 'OppO_SOS'
    OppO_Rank = 'OppO_SOS- Rank'
    OppD_SOS = 'OppD_SOS'
    OppD_Rank = 'OppD_SOS- Rank'
    AdjEM_NCSOS = 'AdjEM_NCSOS'
    AdjEM_NCSOS_Rank = 'AdjEM_NCSOS- Rank'
    TeamName = 'TeamName'
    TeamID = 'TeamID'
    values = [Rk, AdjEM, AdjO, AdjD, AdjT, Luck, AdjEM_SOS, OppO_SOS, OppD_SOS,
              AdjEM_NCSOS]

    def __init__(self, file_name='raw/teams.csv'):
        self.file_name = file_name
        self.df = self.load_teams_df()
        self.teams = {}
        # self.teams = self.set_teams_dict()

    def load_teams_df(self):
        df = pd.read_csv(self.file_name)
        return df

    def set_teams_dict(self):
        cols = [self.TeamID, self.TeamName]
        teams = pd.DataFrame(self.df[[cols]]).set_index(self.TeamName)
        teams = teams.to_dict(orient='index')
        return teams

    def add_kp_stats(self, df_kp):
        self.df = pd.merge(df_kp, self.df, how='left', on=self.TeamName)
        self.teams = self.set_teams()

    @staticmethod
    def get_team_input(df, team):
        return np.array(df[df[Teams.TeamName] == team][Teams.values])

    def set_teams(self):
        for team in self.df[self.TeamName].values:
            team_dict = self.df[self.df[Teams.TeamName] == team]
            team_dict = team_dict.to_dict(orient='records')[0]
            tm = Team(team_dict)
            tm.input = self.get_team_input(self.df, team)
            self.teams[team] = tm
        return self.teams


class Season(object):
    Season = 'Season'
    DayNum = 'DayNum'
    WTeamID = 'WTeamID'
    WScore = 'WScore'
    LTeamID = 'LTeamID'
    LScore = 'LScore'
    WLoc = 'WLoc'
    NumOT = 'NumOT'
    WFGM = 'WFGM'
    WFGA = 'WFGA'
    WFGM3 = 'WFGM3'
    WFGA3 = 'WFGA3'
    WFTM = 'WFTM'
    WFTA = 'WFTA'
    WOR = 'WOR'
    WDR = 'WDR'
    WAst = 'WAst'
    WTO = 'WTO'
    WStl = 'WStl'
    WBlk = 'WBlk'
    WPF = 'WPF'
    LFGM = 'LFGM'
    LFGA = 'LFGA'
    LFGM3 = 'LFGM3'
    LFGA3 = 'LFGA3'
    LFTM = 'LFTM'
    LFTA = 'LFTA'
    LOR = 'LOR'
    LDR = 'LDR'
    LAst = 'LAst'
    LTO = 'LTO'
    LStl = 'LStl'
    LBlk = 'LBlk'
    LPF = 'LPF'

    def __init__(self, year=None, model=None, teams=None,
                 file_name='raw/Prelim2019_RegularSeasonDetailedResults.csv'):
        self.year = year
        self.file_name = file_name
        self.model = model
        self.teams = teams
        self.df = self.load_season_details()
        if teams:
            self.add_teams()

    def load_season_details(self):
        df = pd.read_csv(self.file_name)
        if self.year:
            df = df[df[self.Season] == int(self.year)]
        return df

    def add_teams(self):
        for pre in ['W', 'L']:
            tdf = self.teams.df[:].copy()
            tdf.columns = ['{}{}'.format(pre, col) for col in tdf.columns]
            merge_col = '{}{}'.format(pre, Teams.TeamID)
            self.df = pd.merge(self.df, tdf, how='left', on=merge_col)
        self.df = self.df.dropna()

    def model_season(self):
        y_cols = [self.WScore]
        values = [Teams.values][0]
        x_cols = (['W{}'.format(col) for col in values] +
                  ['L{}'.format(col) for col in values])
        self.model = md.Model(self.df, self.model, x_cols, y_cols)
        self.simulate_season()

    def simulate_season(self):
        team_names = self.teams.df[Teams.TeamName].values
        games = [x for x in itertools.combinations(team_names, 2)]
        for game in games:
            try:
                scores = self.model.score_predictor(game[0], game[1], self.teams)
                game_dict = {game[0]: scores[0],
                             game[1]: scores[1]}
                self.teams.teams[game[0]].games[game[1]] = game_dict
                self.teams.teams[game[1]].games[game[0]] = game_dict
            except:
                continue
