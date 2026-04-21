import os
from dotenv import load_dotenv

load_dotenv()

def get_path(PATH:str):
    raw_relative_path=os.environ[PATH]
    cur_dir=os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.abspath(cur_dir,raw_relative_path))