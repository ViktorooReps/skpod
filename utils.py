import logging
import subprocess
from enum import Enum
from os import PathLike
from pathlib import Path
from time import sleep
from typing import Iterable, List


logger = logging.Logger(__name__)


class NoOutputException(Exception):
    pass


class Machine(Enum):
    POLUS = 'polus'
    BLUEGENE = 'bluegene'


def xlc_compile(src_filename: PathLike, args: Iterable[str] = tuple()):
    args: List[str] = ['xlc'] + list(args) + [str(src_filename)]
    subprocess.run(args)


def schedule(machine: Machine, n_processes: int, exec_file: PathLike, res_filename: str):
    logger.info(f'Scheduling {res_filename} job for {n_processes} processes...')

    if machine == Machine.POLUS:
        args = ['mpisubmit.pl',
                '--processes', str(n_processes),
                '--stdout', res_filename + '.out',
                '--stderr', res_filename + '.err',
                str(exec_file),
                '--', n_processes]
    elif machine == Machine.BLUEGENE:
        args = ['mpisubmit.bg',
                '--nproc', str(n_processes),
                '--stdout', res_filename + '.out',
                '--stderr', res_filename + '.err',
                str(exec_file),
                '--', n_processes]
    else:
        raise ValueError

    subprocess.run(args)

    logger.info(f'Job {res_filename} scheduled!')


def wait(res_filename: str):
    out_file = Path(res_filename + '.out')
    err_file = Path(res_filename + '.err')

    logger.info(f'Waiting for {res_filename} job...')

    while not (out_file.exists() or err_file.exists()):
        sleep(1)

    logger.info(f'Job {res_filename} finished!')


def collect_results(res_filename: str) -> str:
    with open(res_filename + '.out') as f:
        res = f.read()

    if '<OUTPUT>' in res:
        return res.split('<OUTPUT>')[1]
    else:
        with open(res_filename + '.err') as f:
            err_text = f.read()

        raise NoOutputException(err_text)


def save_results(dest: PathLike, results: Iterable[str]):
    combined_results = '\n'.join(results)
    with open(dest, 'w') as f:
        f.write(combined_results)

    logger.info(f'Saved {len(combined_results.splitlines())} entries')
