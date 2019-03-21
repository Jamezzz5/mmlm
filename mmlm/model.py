import logging
import numpy as np
import sklearn.multioutput as mo
import sklearn.linear_model as lm
import sklearn.model_selection as ms
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

log = logging.getLogger()


class Model(object):
    def __init__(self, df, reg, x_cols, y_cols, test=False):
        self.df = df
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.test = test
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.reg = self.set_regressor(reg)
        self.fit = self.regression_fit()
        self.score = self.set_fit_score()

    @staticmethod
    def set_regressor(reg):
        if reg == 'RandomForest':
            reg = RandomForestRegressor()
        if reg == 'Linear':
            reg = lm.LinearRegression()
        if reg == 'BayesianRidge':
            reg = mo.MultiOutputRegressor(lm.BayesianRidge())
        if reg == 'MLP':
            reg = MLPRegressor()
        return reg

    def set_train_values(self):
        self.x_train = self.df[self.x_cols]
        self.y_train = self.df[self.y_cols]
        if self.test:
            self.x_train, self.x_test, self.y_train, self.y_test = \
                ms.train_test_split(self.x_train, self.y_train, test_size=.35,
                                    random_state=0)

    def regression_fit(self):
        logging.info('Fitting data to {}'.format(self.reg))
        self.set_train_values()
        fit = self.reg
        fit.fit(self.x_train, self.y_train)
        return fit

    def set_fit_score(self):
        score = None
        if self.test:
            score = self.fit.score(self.x_test, self.y_test)
            logging.info('Score of fit {} is {}.'.format(self.reg, score))
        return score

    def score_predictor(self, team_name_1, team_name_2, teams):
        team_1 = teams.teams[team_name_1].input
        team_2 = teams.teams[team_name_2].input
        inpt = np.append(team_1, team_2)
        input_val = self.fit.predict([inpt])
        score_1 = input_val[0][0]
        score_2 = input_val[0][1]
        logging.info('{} Score: {}'.format(team_name_1, score_1))
        logging.info('{} Score: {}'.format(team_name_2, score_2))
        p = self.pythagorean_expectation(score_1, score_2)
        return score_1, score_2, p, 1.0 - p

    @staticmethod
    def pythagorean_expectation(s1, s2):
        exponent = 13.91
        pred = (s1 ** exponent) / ((s1 ** exponent) + (s2 ** exponent))
        return pred
