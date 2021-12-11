import logging
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from time import sleep


logger = logging.getLogger(__name__)


class NoOutputException(Exception):
    pass


class Machine:
    POLUS = 'polus'
    BLUEGENE = 'bluegene'


def mpixlc_compile(src_filename, args):
    compiler = 'mpixlc'
    args = [compiler] + list(args) + [str(src_filename)]

    logger.info('Compiling ' + src_filename + ' with ' + str(args) + ' arguments...')
    res = os.system(' '.join(args))
    logger.info('Compilation finished ' + str(res))


def xlc_compile(src_filename, args):
    compiler = 'xlc_r'
    args = [compiler] + list(args) + [str(src_filename)]

    logger.info('Compiling ' + src_filename + ' with ' + str(args) + ' arguments...')
    res = os.system(' '.join(args))
    logger.info('Compilation finished ' + str(res))


def schedule(machine, n_processes, exec_file, res_filename, use_threads=False):
    logger.info('Scheduling ' + res_filename + ' job for ' + str(n_processes)
                + (" threads" if use_threads else " processes") + ' ...')

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

    os.system(' '.join(args))

    logger.info('Job ' + res_filename + ' scheduled!')


def wait(res_filename):
    out_file = res_filename + '.out'
    err_file = res_filename + '.err'

    logger.info('Waiting for ' + res_filename + ' job...')

    finished = False
    while not finished:
        sleep(3)
        if os.path.isfile(out_file):
            with open(out_file) as f:
                contents = f.read()
            finished = finished or len(contents.split('<OUTPUT>')) == 3

        if os.path.isfile(err_file):
            with open(err_file) as f:
                contents = f.read()
            finished = finished or len(contents) > 0

    logger.info('Job ' + res_filename + ' finished!')


def collect_results(res_filename):
    with open(res_filename + '.out') as f:
        res = f.read()

    if '<OUTPUT>' in res:
        return res.split('<OUTPUT>')[1]
    else:
        with open(res_filename + '.err') as f:
            err_text = f.read()

        raise NoOutputException(err_text)


def save_results(dest, results):
    combined_results = ''.join(results)
    with open(dest, 'w') as f:
        f.write(combined_results)

    logger.info('Saved ' + str(len(combined_results.splitlines())) + ' entries')


def create_parser():
    curr_datetime = datetime.now()

    curr_month = curr_datetime.strftime("%b")
    curr_day = curr_datetime.day
    curr_hour = curr_datetime.hour
    curr_minute = curr_datetime.minute

    experiment_desc = str(curr_day) + '-' + curr_month + '-' + str(curr_hour) + ':' + str(curr_minute) + '-job'

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
    parser.add_argument('--mpi', dest='mpi', action='store_true',
                        help='Use MPI to parallel')

    return parser
