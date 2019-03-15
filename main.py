import sys
import logging
import argparse
import mmlm.scraper as scr
import mmlm.basketball as bb

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
    ih = scr.ImportHandler(args.year)
    if args.pull:
        ih.scrape_website_to_df()
    teams = bb.Teams()
    teams.add_kp_stats(ih.df)
    szn = bb.Season(year=args.year)
    szn.add_teams(teams.df)


if __name__ == '__main__':
    main()
