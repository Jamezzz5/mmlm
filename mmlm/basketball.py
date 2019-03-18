import numpy as np
import pandas as pd


class Team(object):
    def __init__(self, team_dict):
        for k in team_dict:
            setattr(self, k, team_dict[k])


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
        for team in self.df[self.Team].values:
            team_dict = self.df[self.df[Teams.TeamName] == team].to_dict(orient='records')[0]
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

    def __init__(self, year=None,
                 file_name='raw/regularseasondetailedresults.csv'):
        self.year = year
        self.file_name = file_name
        self.df = self.load_season_details()

    def load_season_details(self):
        df = pd.read_csv(self.file_name)
        if self.year:
            df = df[df[self.Season] == int(self.year)]
        return df

    def add_teams(self, team_df):
        for pre in ['W', 'L']:
            tdf = team_df[:].copy()
            tdf.columns = ['{}{}'.format(pre, col) for col in tdf.columns]
            merge_col = '{}{}'.format(pre, Teams.TeamID)
            self.df = pd.merge(self.df, tdf, how='left', on=merge_col)
        self.df = self.df.dropna()
