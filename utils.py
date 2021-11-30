import logging
import subprocess
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from enum import Enum
from os import PathLike
from pathlib import Path
from time import sleep
from typing import Iterable, List


logger = logging.getLogger(__name__)


class NoOutputException(Exception):
    pass


class Machine(Enum):
    POLUS = 'polus'
    BLUEGENE = 'bluegene'


def xlc_compile(src_filename: PathLike, args: Iterable[str] = tuple()):
    logger.info(f'Compiling {src_filename}...')

    args: List[str] = ['xlc'] + list(args) + [str(src_filename)]
    subprocess.run(args)

    logger.info(f'Compilation finished')


def schedule(machine: Machine, n_processes: int, exec_file: PathLike, res_filename: str, use_threads: bool = False):
    logger.info(f'Scheduling {res_filename} job for {n_processes} processes...')

    if machine == Machine.POLUS:
        args = ['mpisubmit.pl',
                '--processes', str(n_processes) if not use_threads else '1',
                '--stdout', res_filename + '.out',
                '--stderr', res_filename + '.err',
                str(exec_file),
                '--', str(n_processes)]
    elif machine == Machine.BLUEGENE:
        args = ['mpisubmit.bg',
                '--nproc', str(n_processes) if not use_threads else '1',
                '--stdout', res_filename + '.out',
                '--stderr', res_filename + '.err',
                str(exec_file),
                '--', str(n_processes)]
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


def create_parser() -> ArgumentParser:
    curr_datetime = datetime.now()

    curr_month = curr_datetime.strftime("%b")
    curr_day = curr_datetime.day
    curr_hour = curr_datetime.hour
    curr_minute = curr_datetime.minute

    experiment_desc = f'{curr_day}-{curr_month}-{curr_hour}:{curr_minute}-job'

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('processes', type=int, nargs='+',
                        help='Number of allocated processes')
    parser.add_argument('--src_file', type=str,
                        help='Path to source file')
    parser.add_argument('--exec_file', type=str, default='exec',
                        help='Name of executable after compilation')
    parser.add_argument('--experiment', type=str, default=experiment_desc,
                        help='Experiment name')
    parser.add_argument('--results', type=str, default='results.csv',
                        help='Filename to save results to')
    parser.add_argument('--mpi', type=bool, default=False,
                        help='Use MPI to parallel')

    return parser
