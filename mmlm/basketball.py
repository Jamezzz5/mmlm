import os
import random
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
        self.normalize_x()
        self.teams = self.set_teams()

    def normalize_x(self):
        for col in self.values:
            min_v = min(self.df[col])
            max_v = max(self.df[col])
            self.df[col] = ((self.df[col] - min_v) / (max_v - min_v))

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
    AScore = 'AScore'
    BScore = 'BScore'

    def __init__(self, year=None, model_name=None, teams=None,
                 file_name='raw/Prelim2019_RegularSeasonDetailedResults.csv'):
        self.year = year
        self.file_name = file_name
        self.model_name = model_name
        self.teams = teams
        self.model = None
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

    def randomize_cols(self, values, seed=False):
        if seed:
            np.random.seed(0)
        self.df['rnd'] = np.random.randint(2, size=len(self.df.index))
        cd = {}
        for col in values:
            for x in ['A', 'B', 'W', 'L']:
                cd[x] = '{}{}'.format(x, col)
            mask = self.df['rnd'] == 1
            self.df[cd['A']] = np.where(mask,
                                        self.df[cd['W']], self.df[cd['L']])
            self.df[cd['B']] = np.where(mask,
                                        self.df[cd['L']], self.df[cd['W']])
        return self.df

    def model_season(self):
        self.set_season_model()
        self.simulate_season()

    def set_season_model(self):
        cols = Teams.values
        self.df = self.randomize_cols(values=cols + ['Score'])
        y_cols = [self.AScore, self.BScore]
        x_cols = (['A{}'.format(col) for col in cols] +
                  ['B{}'.format(col) for col in cols])
        self.model = md.Model(self.df, self.model_name, x_cols, y_cols, test=True)

    def simulate_season(self):
        team_names = self.teams.df[Teams.TeamName].values
        np.random.shuffle(team_names)
        games = [x for x in itertools.combinations(team_names, 2)]
        random.shuffle(games)
        for game in games:
            s1, s2, p1, p2 = self.model.score_predictor(game[0], game[1],
                                                        self.teams)
            game_dict = {game[0]: s1, game[1]: s2, 'p': p1}
            self.teams.teams[game[0]].games[game[1]] = game_dict
            game_dict = {game[0]: s1, game[1]: s2, 'p': p2}
            self.teams.teams[game[1]].games[game[0]] = game_dict
        self.generate_submission_file()

    def generate_submission_file(self):
        self.teams.df[Teams.TeamID] = (self.teams.df[Teams.TeamID].fillna(0).
                                       astype('int'))
        team_map = self.teams.df[[Teams.TeamName, Teams.TeamID]]
        team_map = team_map.set_index([Teams.TeamID]).to_dict(orient='dict')
        df = pd.read_csv('raw/SamplesubmissionStage2.csv')
        df = df.join(df['ID'].str.split('_', expand=True))
        for col in [1, 2]:
            df[col] = df[col].astype('int').map(team_map[Teams.TeamName])
        match_dict = df[[1, 2]].to_dict('index')
        for x in match_dict:
            team_1 = match_dict[x][1]
            team_2 = match_dict[x][2]
            df.iloc[x, 1] = self.teams.teams[team_1].games[team_2]['p']
        file_name = '{}_{}.csv'.format(self.year, self.model_name)
        file_name = os.path.join('pred', file_name)
        df[['ID', 'Pred']].to_csv(file_name, index=False)


class Game(object):
    def __init__(self):
        pass
