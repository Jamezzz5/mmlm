import requests
import logging
import os
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import math


def dir_check(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


class WebTable(object):
    def __init__(self, url):
        logging.info('Finding tables at ' + url)
        self.url = url
        self.df = pd.DataFrame()
        r = requests.get(self.url)
        self.column_names = []
        self.soup = BeautifulSoup(r.text, 'lxml')
        self.soup.find_all('table id')
        self.table_body = None
        self.table_head = None
        self.rows = None

    def get_df(self):
        return self.df

    def get_table_body(self):
        self.table_body = self.soup.find('tbody')

    def get_table_head(self):
        self.table_head = self.soup.find('thead')

    def get_table_rows(self, table_part):
        self.rows = table_part.find_all('tr')

    def headers_to_df(self):
        logging.info('Reading table headers to df.')
        self.get_table_head()
        self.get_table_rows(self.table_head)
        cols = self.rows[1].find_all('th')
        x = 0
        for col in cols:
            col = col.text.strip()
            self.column_names.append(col)
            if x > 4:
                col += '- Rank'
                self.column_names.append(col)
            x += 1
        self.df.columns = self.column_names

    def body_to_df(self):
        logging.info('Reading table body to df.')
        self.get_table_body()
        self.get_table_rows(self.table_body)
        for row in self.rows:
            cols = row.find_all('td')
            cols = [x.text.strip() for x in cols]
            if cols:
                cols = pd.Series(cols)
                self.df = self.df.append(cols, ignore_index=True)

    def df_remove_numbers(self):
        self.df['Team'] = self.df['Team'].str.replace('\d+', '').str.strip()

    def df_remove_period(self):
        self.df['Team'] = self.df['Team'].str.replace('.', '')

    def team_translation(self):
        tt = pd.read_csv('raw/team_translate.csv')
        self.df = self.df.merge(tt, on='Team', how='left')
        self.df['Team_Name'] = np.where(self.df['Team_Name'].isnull(),
                                        self.df['Team'], self.df['Team_Name'])

    def table_to_df(self):
        self.body_to_df()
        self.headers_to_df()
        self.df_remove_numbers()
        self.df_remove_period()
        self.team_translation()

    def df_to_csv(self, file_path, file_name):
        logging.info('Writing df to ' + file_path + '/' + file_name)
        dir_check(file_path)
        self.df.to_csv(file_path + '/' + file_name + '.csv', index=False)
