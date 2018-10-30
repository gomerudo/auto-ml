import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(CURRENT_DIR, '../..')
ROOT_DIR = os.path.abspath(ROOT_DIR)
sys.path.append(ROOT_DIR)