import logging
import numpy as np
import sklearn.linear_model as lm
from sklearn.ensemble import RandomForestRegressor

log = logging.getLogger()


class Model(object):
    def __init__(self, df, clf, x_cols, y_cols):
        self. df = df
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.clf = self.set_classifier(clf)
        self.fit = self.regression_fit()

    def set_classifier(self, clf):
        if clf == 'RandomForest':
            clf = RandomForestRegressor
        if clf == 'Linear':
            clf = lm.LinearRegression
        if clf == 'BayesianRidge':
            clf = lm.BayesianRidge
        return clf

    def get_team_input(self, df, team):
        return np.array(df[df['TeamName'] == team][values])

    def regression_fit(self):
        x = self.df[self.x_cols]
        y = self.df[self.y_cols]
        fit = self.clf(n_estimators=1000)
        fit.fit(x, y)
        return fit

    def score_predictor(self, team_name_1, team_name_2):
        team_1 = self.get_team_input(df_kp, team_name_1)
        team_2 = self.get_team_input(df_kp, team_name_2)
        input_1 = np.append(team_1, team_2)
        input_2 = np.append(team_2, team_1)
        input_1_val = self.fit.predict([input_1])
        input_2_val = self.fit.predict([input_2])
        logging.info('{} Score: {}'.format(team_name_1, input_1_val))
        logging.info('{} Score: {}'.format(team_name_2, input_2_val))
        return [input_1_val, input_2_val]


    clf = regression_fit(x_values, y_values)
    score_predictor('Michigan', 'Louisville')
