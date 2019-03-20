import logging
import numpy as np
import sklearn.linear_model as lm
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

log = logging.getLogger()


class Model(object):
    def __init__(self, df, clf, x_cols, y_cols):
        self.df = df
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.x_vals = None
        self.y_vals = None
        self.reg = self.set_regressor(clf)
        self.fit = self.regression_fit()

    @staticmethod
    def set_regressor(reg):
        if reg == 'RandomForest':
            reg = RandomForestRegressor
        if reg == 'Linear':
            reg = lm.LinearRegression
        if reg == 'BayesianRidge':
            reg = lm.BayesianRidge
        if reg == 'MLP':
            reg = MLPRegressor
        return reg

    def regression_fit(self):
        self.x_vals = self.df[self.x_cols]
        self.y_vals = self.df[self.y_cols]
        fit = self.reg()
        fit.fit(self.x_vals, self.y_vals)
        return fit

    def score_predictor(self, team_name_1, team_name_2, teams):
        team_1 = teams.teams[team_name_1].input
        team_2 = teams.teams[team_name_2].input
        input_1 = np.append(team_1, team_2)
        input_2 = np.append(team_2, team_1)
        input_1_val = self.fit.predict([input_1])
        input_2_val = self.fit.predict([input_2])
        logging.info('{} Score: {}'.format(team_name_1, input_1_val))
        logging.info('{} Score: {}'.format(team_name_2, input_2_val))
        return [input_1_val, input_2_val]
