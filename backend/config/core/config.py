import os
from dotenv import load_dotenv

load_dotenv()

def get_path(PATH:str) ->str:
    # Get the relative path from env
    raw_relative_path=os.environ[PATH]
    # get the absolute path of this python script directory
    cur_dir=os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.abspath(os.path.join(cur_dir,raw_relative_path)))