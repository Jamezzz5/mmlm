import sys
import logging
import argparse
import numpy as np
import pandas as pd
import mmlm.scraper as scr
from sklearn.ensemble import RandomForestRegressor

formatter = logging.Formatter('%(asctime)s [%(module)14s]'
                              '[%(levelname)8s] %(message)s')
log = logging.getLogger()
log.setLevel(logging.INFO)

console = logging.StreamHandler(sys.stdout)
console.setFormatter(formatter)
log.addHandler(console)

log_file = logging.FileHandler('logfile.log', mode='w')
log_file.setFormatter(formatter)
log.addHandler(log_file)

parser = argparse.ArgumentParser()
parser.add_argument('--year', metavar='N', type=int)
parser.add_argument('--pull', action='store_true')
args = parser.parse_args()


def main():
    pass


if __name__ == '__main__':
    main()

year = '2018'
url_kp = 'http://kenpom.com/index.php?y={}'.format(year)
table = scr.WebTable(url_kp)
table.table_to_df()
filename_kp = url_kp.replace('http://', '').replace('.com/index.php?y=', '')
filename_kp += '.csv'
table.df_to_csv('raw', filename_kp)

df_kp = pd.read_csv('raw/' + filename_kp)
df_teams = pd.read_csv('raw/teams.csv')

df = pd.merge(df_kp, df_teams, how='left', on='TeamName')

df_result = pd.read_csv('raw/regularseasondetailedresults.csv')
df_result_year = df_result[df_result['Season'] == int(year)]

df_teams_w = df[:].copy()
df_teams_w.columns = ['W' + str(col) for col in df_teams_w.columns]
df_teams_l = df[:].copy()
df_teams_l.columns = ['L' + str(col) for col in df_teams_l.columns]


df_result_year = pd.merge(df_result_year, df_teams_w, how='left',
                          left_on='WTeamID', right_on='WTeamID')
df_result_year = pd.merge(df_result_year, df_teams_l, how='left',
                          left_on='LTeamID', right_on='LTeamID')

df_result_year_w = df_result_year[:].copy()
df_result_year_w['Score'] = df_result_year_w['WScore']
df_result_year_w['Team'] = df_result_year_w['WTeam']
df_result_year_l = df_result_year[:].copy()
df_result_year_l['Score'] = df_result_year_l['LScore']
df_result_year_l['Team'] = df_result_year_l['LTeam']
df_final_result_year = pd.DataFrame(columns=df_result_year_w.columns)
df_final_result_year = df_final_result_year.append(df_result_year_w)
df_final_result_year = df_final_result_year.append(df_result_year_l)

df_final_result_year = df_final_result_year.dropna()

y_values = ['Score']

values = ['AdjEM', 'AdjO', 'AdjD', 'AdjT', 'Luck', 'AdjEM.1', 'OppO', 'OppD']
x_values = ['W' + val for val in values] + ['L' + val for val in values]


def get_team_input(df, team):
    return np.array(df[df['TeamName'] == team][values])


def regression_fit(x_values, y_values,
                   model=RandomForestRegressor):
    x = df_final_result_year[x_values]
    y = df_final_result_year[y_values]
    model_fit = model(n_estimators=1000)
    model_fit.fit(x, y)
    return model_fit


def score_predictor(clf, team_name_1, team_name_2):
    team_1 = get_team_input(df_kp, team_name_1)
    team_2 = get_team_input(df_kp, team_name_2)
    input_1 = np.append(team_1, team_2)
    input_2 = np.append(team_2, team_1)
    input_1_val = clf.predict([input_1])
    input_2_val = clf.predict([input_2])
    logging.info('{} Score: {}'.format(team_name_1, input_1_val))
    logging.info('{} Score: {}'.format(team_name_2, input_2_val))
    return [input_1_val, input_2_val]


clf = regression_fit(x_values, y_values)
score_predictor(clf, 'Michigan', 'Louisville')

team_loss_dict = {}
teams = df_kp['Team'].values
for team in teams:
    try:
        losses = [x for x in df_kp['TeamName'] if
                  score_predictor(clf, team, x)[0] <
                  score_predictor(clf, team, x)[1]]
        team_loss_dict[team] = losses
    except:
        continue


from operator import itemgetter
d = team_loss_dict
d = {x: len(d[x]) for x in d}
sorted(d.items(), key=itemgetter(1))
