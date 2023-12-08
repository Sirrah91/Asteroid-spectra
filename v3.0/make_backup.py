from shutil import copytree, copy2, make_archive
from glob import glob
from os import path

from modules.utilities import check_dir
from modules._constants import _path_backup, _project_dir, _subdirs


# pipreqs --encoding utf-8 --ignore venv10,backup ./


def make_backup(version: str) -> None:
    # definition of backup dirs and creating them
    backup_dir = path.join(_path_backup, version)

    if path.isdir(backup_dir):
        raise ValueError('The back-up dir already exists. Change "version" parameter.')

    # directories in _project_dir
    dirs_to_save_specific = {_subdirs["modules"]: ["*.py"],
                             path.join(_subdirs["HP_tuning"], "Bayes"): ["*.csv"],
                             path.join(_subdirs["HP_tuning"], "Random"): ["*.csv"],
                             _subdirs["web_app"]: ["*.py", "requirements.txt"],
                             path.join(_subdirs["web_app"], "support"): ["*.npz"],
                             path.join(_subdirs["web_app"], "templates"): ["*.html"],
                             "": ["ma*.py", "requirements*.txt"],  # empty folder can't be first in the dict
                             }

    # directories in _project_dir
    dirs_to_save_all = {_subdirs["datasets"],
                        path.join(_subdirs["models"], "composition"),
                        path.join(_subdirs["models"], "taxonomy"),
                        _subdirs["accuracy_tests"],
                        _subdirs["asteroid_images"],
                        }

    # save specific suffixes
    for directory, suffixes in dirs_to_save_specific.items():
        dir_to_backup = path.join(_project_dir, directory)

        if not path.isdir(dir_to_backup):
            print(f'Directory "{dir_to_backup}" does not exist. Skipping it...')
            continue

        backup_dir_name = path.join(backup_dir, directory)
        check_dir(backup_dir_name)

        for suffix in suffixes:
            source = path.join(dir_to_backup, suffix)

            for file in glob(source):
                copy2(file, backup_dir_name)

    # save all that is inside
    for directory in dirs_to_save_all:
        source = path.join(_project_dir, directory)

        if not path.isdir(source):
            print(f'Directory "{source}" does not exist. Skipping it...')
            continue

        copytree(source, path.join(backup_dir, directory))  # copytree creates the folder automatically

    # zip the folder
    make_archive(backup_dir, "zip", backup_dir)


if __name__ == "__main__":
    make_backup("v3.4.8")
