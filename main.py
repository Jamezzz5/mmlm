import sys
import logging
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


url_kp = 'http://kenpom.com/index.php?y=2016'
table = scr.WebTable(url_kp)
table.table_to_df()
filename_kp = url_kp.replace('http://', '').replace('.com/index.php?y=','')
table.df_to_csv('raw', filename_kp)
