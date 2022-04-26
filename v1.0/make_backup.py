from modules.utilities import check_dir
from shutil import copy2, make_archive
from glob import glob


project_dir = '/home/dakorda/Python/NN/'
backup_dir_pref = '/home/dakorda/Python/NN/backup/'


def make_backup(version: str) -> None:
    # definition of backup dirs and creating them
    backup_dir = "".join((backup_dir_pref, version))
    backup_dir_modules = "".join((backup_dir, '/modules/'))
    backup_dir_RELAB = "".join((backup_dir, '/Datasets/RELAB/'))
    backup_dir_CTape = "".join((backup_dir, '/Datasets/C-Tape/'))
    backup_dir_Tuomas = "".join((backup_dir, '/Datasets/Tuomas/'))
    backup_dir_Tomas_mix = "".join((backup_dir, '/Datasets/mix_test/'))
    backup_dir_Tomas_met = "".join((backup_dir, '/Datasets/met_test/'))
    backup_dir_Datasets = "".join((backup_dir, '/Datasets/'))
    backup_dir_models = "".join((backup_dir, '/Models/chemical/'))

    check_dir(backup_dir_modules)
    check_dir(backup_dir_RELAB)
    check_dir(backup_dir_CTape)
    check_dir(backup_dir_Tuomas)
    check_dir(backup_dir_Tomas_mix)
    check_dir(backup_dir_Tomas_met)
    check_dir(backup_dir_Datasets)
    check_dir(backup_dir_models)

    # copy main and make_backup
    source = "".join((project_dir, '/ma*.py'))
    for file in glob(source):
        copy2(file, backup_dir)

    # copy modules
    source = "".join((project_dir, '/modules/*.py'))
    for file in glob(source):
        copy2(file, backup_dir_modules)

    # copy datasets
    source = "".join((project_dir, '/Datasets/RELAB/*.dat'))
    for file in glob(source):
        copy2(file, backup_dir_RELAB)
    source = "".join((project_dir, '/Datasets/RELAB/*.xlsx'))
    for file in glob(source):
        copy2(file, backup_dir_RELAB)

    source = "".join((project_dir, '/Datasets/Tuomas/*.dat'))
    for file in glob(source):
        copy2(file, backup_dir_Tuomas)
    source = "".join((project_dir, '/Datasets/Tuomas/*.h5'))
    for file in glob(source):
        copy2(file, backup_dir_Tuomas)
    source = "".join((project_dir, '/Datasets/Tuomas/*.txt'))
    for file in glob(source):
        copy2(file, backup_dir_Tuomas)

    source = "".join((project_dir, '/Datasets/mix_test/*'))
    for file in glob(source):
        copy2(file, backup_dir_Tomas_mix)

    source = "".join((project_dir, '/Datasets/met_test/*'))
    for file in glob(source):
        copy2(file, backup_dir_Tomas_met)

    source = "".join((project_dir, '/Datasets/C-Tape/*'))
    for file in glob(source):
        copy2(file, backup_dir_CTape)

    source = "".join((project_dir, '/Datasets/*.dat'))
    for file in glob(source):
        copy2(file, backup_dir_Datasets)

    # copy models
    source = "".join((project_dir, '/Models/chemical/*.h5'))
    for file in glob(source):
        copy2(file, backup_dir_models)

    # zip the folder
    # make_archive(backup_dir, 'zip', backup_dir)


if __name__ == '__main__':
    make_backup('v1.2')
