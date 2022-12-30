from shutil import copytree, copy2, make_archive
from glob import glob
from os import path

from modules.utilities import check_dir

from modules._constants import _path_backup, _project_dir


def make_backup(version: str) -> None:
    # definition of backup dirs and creating them
    backup_dir = "".join((_path_backup, version))

    dirs_to_save_specific = {"modules": ["*.py"],
                             "Models/compositional": ["*.h5"],
                             "Models/taxonomical": ["*.h5"],
                             "tuning_HP/Bayes": ["*.csv"],
                             "tuning_HP/Random": ["*.csv"],
                             "": ["ma*.py"]}  # empty folder can't be first in the dict

    dirs_to_save_all = {"Datasets",
                        "accuracy_test",
                        "Asteroid_images"}

    # save specific suffixes
    for directory, suffixes in dirs_to_save_specific.items():
        dir_to_backup = "".join((_project_dir, "/", directory, "/"))

        if not path.isdir(dir_to_backup):
            print('Directory "{dir_name:s}" does not exist. Skipping it...'.format(dir_name=dir_to_backup))
            continue

        backup_dir_name = "".join((backup_dir, "/", directory, "/"))
        check_dir(backup_dir_name)

        for suffix in suffixes:
            source = "".join((dir_to_backup, suffix))

            for file in glob(source):
                copy2(file, backup_dir_name)

    # save all what is inside
    for directory in dirs_to_save_all:
        source = "".join((_project_dir, "/", directory, "/"))

        if not path.isdir(source):
            print('Directory "{dir_name:s}" does not exist. Skipping it...'.format(dir_name=source))
            continue

        copytree(source, "".join((backup_dir, "/", directory, "/")))  # copytree creates the folder automatically

    # zip the folder
    make_archive(backup_dir, "zip", backup_dir)


if __name__ == "__main__":
    make_backup("v2.1.1")
