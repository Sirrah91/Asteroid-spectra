from modules.utilities import check_dir
from shutil import copy2
from glob import glob

project_dir = '/home/dakorda/Python/NN/'
backup_dir_pref = '/home/dakorda/Python/NN/backup/'


def make_backup(version: str, ) -> None:
    # definition of backup dirs and creating them
    backup_dir = "".join((backup_dir_pref, version, '/'))
    backup_dir_modules = "".join((backup_dir, '/modules/'))
    check_dir(backup_dir_modules)

    # copy main and make_backup
    source = "".join((project_dir, '/ma*.py'))
    for file in glob(source):
        copy2(file, backup_dir)

    # copy modules
    source = "".join((project_dir, '/modules/*.py'))
    for file in glob(source):
        copy2(file, backup_dir_modules)


if __name__ == '__main__':
    make_backup('v1.1')
