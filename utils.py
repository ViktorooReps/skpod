import logging
import subprocess
from enum import Enum
from io import StringIO
from os import PathLike
from pathlib import Path
from time import sleep
from typing import Iterable, List

import pandas as pd
from pandas import DataFrame


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
        args = ['mpisubmit.pl',
                '--processes', str(n_processes),
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


def collect_results(res_filename: str) -> DataFrame:
    with open(res_filename + '.out') as f:
        res = f.read()

    if '<OUTPUT>' in res:
        csv_text = res.split('<OUTPUT>')[1]

        return pd.read_csv(StringIO(csv_text), sep='\t')
    else:
        with open(res_filename + '.err') as f:
            err_text = f.read()

        raise NoOutputException(err_text)


def save_results(dest: PathLike, results: Iterable[DataFrame]):
    concat_df = pd.concat(tuple(results))
    concat_df.to_csv(str(dest), sep='\t', index=False)

    logger.info(f'Saved {len(concat_df.index)} entries')
