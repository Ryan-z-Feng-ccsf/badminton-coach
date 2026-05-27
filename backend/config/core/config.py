import os
import typing
from dotenv import load_dotenv
import uuid
import shutil

load_dotenv()


def get_path(path: str) -> str:
    # Get the relative path from env
    raw_relative_path = os.environ[path]
    # get the absolute path of this python script directory
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.join(cur_dir,"../../"))
    return os.path.abspath(os.path.join(project_root, raw_relative_path))


class TempVideoManager:
    def __init__(self, temp_video_path: str):

        self._saved_path = self._generate_file_name()
        self._source_file:typing.BinaryIO = temp_video_path

    def __enter__(self):

        self._destination_file = open(self._saved_path, "wb")
        shutil.copyfileobj(self._source_file, self._destination_file)
        return self._saved_path

    def __exit__(self, exc_type, exc, tb):
        if self._source_file:
            self._source_file.close()
        if self._destination_file:
            self._destination_file.close()
            if os.path.exists(self._saved_path):
                os.remove(self._saved_path)
        if exc_type:
            print(f"An error occurred: {exc}")
        return False

    def _generate_file_name(self):
        unique_id = str(uuid.uuid4())
        file_path = os.path.abspath(f"data/raw_videos/{unique_id}.mp4")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        return file_path
