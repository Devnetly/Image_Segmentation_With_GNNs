import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUTPUTS_DIR = os.path.join(ROOT_DIR, 'outputs')

PanNuke_DIR = os.path.join(DATA_DIR, 'PanNuke')
ISIC_DIR = os.path.join(DATA_DIR, 'ISBI2016_ISIC_Part3B_Training_Data')
EMDS_6_DIR = os.path.join(DATA_DIR, 'EMDS-6')