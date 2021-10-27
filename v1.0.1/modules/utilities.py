from pathlib import Path


def check_dir(dir_or_file_path: str) -> None:
    dir_or_file_path = Path(dir_or_file_path)

    if dir_or_file_path.suffix:
        directory = dir_or_file_path.parent  # is file
    else:
        directory = dir_or_file_path  # is folder

    if not directory.exists():
        print('Directory ' + directory.as_posix() + " doesn't exist, creating it now.")
        directory.mkdir(parents=True, exist_ok=True)
