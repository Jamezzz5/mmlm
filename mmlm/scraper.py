import os
import logging
import requests
import numpy as np
import pandas as pd
import mmlm.utils as utl
from bs4 import BeautifulSoup


class ImportHandler(object):
    def __init__(self, year):
        self.year = year
        self.file_path = 'raw'
        self.url = self.get_kp_url(self.year)
        self.file_name = self.get_kp_filename(self.year)
        self.full_file_name = os.path.join(self.file_path, self.file_name)
        self.df = pd.DataFrame()
        if os.path.exists(self.full_file_name):
            self.df = pd.read_csv(self.full_file_name)

    @staticmethod
    def get_kp_url(year):
        return 'http://kenpom.com/index.php?y={}'.format(year)

    @staticmethod
    def get_kp_filename(year):
        return 'kenpom{}.csv'.format(year)

    def scrape_website_to_df(self):
        table = WebTable(self.url)
        table.table_to_df()
        table.df_to_csv(file_path='raw', file_name=self.file_name)
        self.df = table.df

    def add_ids_to_df(self, team_id_csv='raw/teams.csv', name_col='TeamName'):
        if self.df.empty:
            self.df = pd.read_csv(os.path.join('raw', self.file_name))
        team_id_df = pd.read_csv(team_id_csv)
        self.df = pd.merge(self.df, team_id_df, how='left', on=name_col)


class WebTable(object):
    def __init__(self, url):
        logging.info('Finding tables at {}'.format(url))
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
        for idx, col in enumerate(cols):
            col = col.text.strip()
            if 8 < idx < 12:
                col += '_SOS'
            if idx == 12:
                col += '_NCSOS'
            self.column_names.append(col)
            if idx > 4:
                col += '- Rank'
                self.column_names.append(col)
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

    def team_translation(self, team_col='TeamName'):
        tt = pd.read_csv('raw/team_translate.csv')
        self.df = self.df.merge(tt, on='Team', how='left')
        self.df[team_col] = np.where(self.df[team_col].isnull(),
                                     self.df['Team'], self.df[team_col])

    def table_to_df(self):
        self.body_to_df()
        self.headers_to_df()
        self.df_remove_numbers()
        self.df_remove_period()
        self.team_translation()

    def df_to_csv(self, file_path, file_name):
        full_file_name = os.path.join(file_path, file_name)
        logging.info('Writing df to {}'.format(full_file_name))
        utl.dir_check(file_path)
        self.df.to_csv(full_file_name, index=False)
