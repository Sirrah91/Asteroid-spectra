from modules.utilities import check_dir
from shutil import copytree, copy2, make_archive
from glob import glob


project_dir = '/home/dakorda/Python/NN/'
backup_dir_pref = '/home/dakorda/Python/NN/backup/'


def make_backup(version: str) -> None:
    # definition of backup dirs and creating them
    backup_dir = "".join((backup_dir_pref, version))
    backup_dir_modules = "".join((backup_dir, '/modules/'))
    backup_dir_models = "".join((backup_dir, '/Models/'))
    backup_dir_datasets = "".join((backup_dir, '/Datasets/'))
    backup_dir_range_test = "".join((backup_dir, '/range_test_data/'))

    check_dir(backup_dir_modules)
    # check_dir(backup_dir_models)  # copytree creates the folder
    # check_dir(backup_dir_datasets)  # copytree creates the folder
    # check_dir(backup_dir_range_test)  # copytree creates the folder

    # copy main* and make_backup
    source = "".join((project_dir, '/ma*.py'))
    for file in glob(source):
        copy2(file, backup_dir)

    # copy modules
    source = "".join((project_dir, '/modules/*.py'))
    for file in glob(source):
        copy2(file, backup_dir_modules)

    # copy models
    source = "".join((project_dir, '/Models/'))
    copytree(source, backup_dir_models)

    # copy datasets
    source = "".join((project_dir, '/Datasets/'))
    copytree(source, backup_dir_datasets)

    source = "".join((project_dir, '/range_test_data/'))
    copytree(source, backup_dir_range_test)

    # zip the folder
    make_archive(backup_dir, 'zip', backup_dir)


if __name__ == '__main__':
    make_backup('v1.1.2')
