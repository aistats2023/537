import os
import subprocess


def is_git_controlled():
    rc = subprocess.call(['git', 'rev-parse', 'HEAD'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return rc == 0


def git_head():
    if is_git_controlled():
        info = subprocess.check_output(['git', 'log', '-1', '--oneline']).strip().decode('utf-8')
    else:
        info = 'No git found at {}'.format(os.getcwd())
    return info


def git_head_to_file(filename):
    info = git_head() + '\n'
    with open(filename, 'w') as fp:
        fp.write(info)