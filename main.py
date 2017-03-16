import sys
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import utils.scraper as scr

logging.basicConfig(stream=sys.stdout,
                    filename='logfile.log',
                    filemode='w',
                    level=logging.INFO,
                    disable_existing_loggers=False,
                    format=('%(asctime)s [%(module)14s]' +
                            '[%(levelname)8s] %(message)s'))
console = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s [%(module)14s]' +
                              '[%(levelname)8s] %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


year = '2017'
url_kp = 'http://kenpom.com/index.php?y={}'.format(year)
table = scr.WebTable(url_kp)
table.table_to_df()
filename_kp = url_kp.replace('http://', '').replace('.com/index.php?y=', '')
filename_kp += '.csv'
table.df_to_csv('raw', filename_kp)

df_kp = pd.read_csv('raw/' + filename_kp)
df_teams = pd.read_csv('raw/teams.csv')

df = pd.merge(df_kp, df_teams, how='left', on='Team_Name')

df_result = pd.read_csv('raw/regularseasondetailedresults.csv')
df_result_year = df_result[df_result['Season'] == int(year)]

df_teams_w = df[:].copy()
df_teams_w.columns = ['W' + str(col) for col in df_teams_w.columns]
df_teams_l = df[:].copy()
df_teams_l.columns = ['L' + str(col) for col in df_teams_l.columns]

df_result_year = pd.merge(df_result_year, df_teams_w, how='left',
                          left_on='Wteam', right_on='WTeam_Id')
df_result_year = pd.merge(df_result_year, df_teams_l, how='left',
                          left_on='Lteam', right_on='LTeam_Id')

df_result_year_w = df_result_year[:].copy()
df_result_year_w['Score'] = df_result_year_w['Wscore']
df_result_year_w['Team'] = df_result_year_w['Wteam']
df_result_year_l = df_result_year[:].copy()
df_result_year_l['Score'] = df_result_year_l['Lscore']
df_result_year_l['Team'] = df_result_year_l['Lteam']
df_final_result_year = pd.DataFrame(columns=df_result_year_w.columns)
df_final_result_year = df_final_result_year.append(df_result_year_w)
df_final_result_year = df_final_result_year.append(df_result_year_l)

df_final_result_year = df_final_result_year.dropna()

y_values = ['Score']

values = ['AdjEM', 'AdjO', 'AdjD', 'AdjT', 'Luck', 'AdjEM.1', 'OppO', 'OppD']
x_values = ['W' + val for val in values] + ['L' + val for val in values]


def get_team_input(df, team):
    return np.array(df[df['Team_Name']==team][values])

def regression_fit(x_values, y_values):
    X = df_final_result_year[x_values]
    y = df_final_result_year[y_values]
    clf = RandomForestRegressor(n_estimators=1000)
    clf.fit(X, y)
    return clf

team_1 = get_team_input(df_kp, 'Michgan')
team_2 = get_team_input(df_kp, 'Oklahoma St')

input_1 = team_1 + team_2

print 'Michigan Score: {}'.format(clf.predict(mich_input).reshape(-1, 1))
print 'OSU Score: {}'.format(clf.predict(osu_wins).reshape(-1, 1))
