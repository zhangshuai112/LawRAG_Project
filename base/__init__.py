import sys
import os

current_path = os.path.abspath(__file__)
base_path = os.path.dirname(current_path)
project_path = os.path.dirname(base_path)
if base_path not in sys.path:
    sys.path.insert(0,base_path)
if project_path not in sys.path:
    sys.path.insert(0,project_path)


from config import Config
from logger import logger