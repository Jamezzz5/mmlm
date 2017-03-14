import sys
import logging
import pandas as pd
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


year = '2016'
url_kp = 'http://kenpom.com/index.php?y={}'.format(year)
table = scr.WebTable(url_kp)
table.table_to_df()
filename_kp = url_kp.replace('http://', '').replace('.com/index.php?y=', '')
table.df_to_csv('raw', filename_kp)

df_kp = pd.read_csv(filename_kp)
df_teams = pd.read_csv('raw/teams.csv')

df = pd.merge(df_kp, df_teams, how='left', on='Team_Name')

df_result = pd.read_csv('raw/tourneydetailedresults.csv')
df_result_year = df_result[df_result['Season'] == int(year)]

df_teams_w = df[:].copy()
df_teams_w.columns = ['W' + str(col) for col in df_teams_w.columns]
df_teams_l = df[:].copy()
df_teams_l.columns = ['L' + str(col) for col in df_teams_l.columns]

df_result_year = pd.merge(df_result_year, df_teams_w, how='left',
                          left_on='Wteam', right_on='WTeam_Id')
df_result_year = pd.merge(df_result_year, df_teams_l, how='left',
                          left_on='Lteam', right_on='LTeam_Id')



y_values = ['Wscore', 'Lscore']
x_values = ['WAdjEM', 'LAdjEM', 'WAdjO', 'LAdjO', 'WAdjD', 'LAdjD', 'WAdjT',
            'LAdjT', 'WLuck', 'LLuck', 'WAdjEM', 'LAdjEM', 'WOppO', 'LOppO',
            'WOppD', 'LOppD']
mich_wins = np.array([22.62, 22.10, 121.7, 124.8, 99.0, 102.7, 62.5, 69.9,
                      -0.031, -0.056, 9.68, 13.97, 109.2, 111.6, 99.5, 97.6])

osu_wins = np.array([22.10, 22.62, 124.8, 121.7, 102.7, 99.0, 69.9, 62.5,
                     -0.056, -0.031, 13.97, 9.68, 111.6, 109.2, 97.6, 99.5])

X = df_result_year[x_values]
for yval in y_values:
    y = df_result_year[yval]
    reg = linear_model.BayesianRidge()
    reg.fit(X, y)
    print 'X Value: {} Y Value: {} COEF: {}'.format(xval, yval, reg.coef_)
    print 'Michigan Wins: {}'.format(reg.predict(mich_wins).reshape(1,-1))
    print 'OSU Wins: {}'.format(reg.predict(osu_wins).reshape(1,-1))

X = df_result_year[x_values]
for yval in y_values:
    y = df_result_year[yval]
    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit(X, y)
    print 'X Value: {} Y Value: {} COEF: {}'.format(xval, yval, clf.score)
    print 'Michigan Wins: {}'.format(clf.predict(mich_wins).reshape(1,-1))
    print 'OSU Wins: {}'.format(clf.predict(osu_wins).reshape(1,-1))