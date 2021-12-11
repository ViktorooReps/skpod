import logging
import os

import sys

from utils import create_parser, xlc_compile, schedule, Machine, wait, collect_results, save_results, mpixlc_compile

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO, stream=sys.stdout)

    arg_parser = create_parser()
    args = arg_parser.parse_args()

    compile_args = ['-qsmp=omp', '-qthreaded', '-lpthread', '-o', args.exec_file]
    if args.mpi:
        mpixlc_compile(args.src_file, compile_args)
    else:
        xlc_compile(args.src_file, compile_args)

    if not os.path.exists(args.exec_file):
        raise ValueError('Compilation failed!')

    results = []
    for n_proc in args.processes:
        job_name = args.experiment + '-' + str(n_proc) + ("p" if args.mpi else "t")

        schedule(Machine.BLUEGENE, n_proc, exec_file=args.exec_file, res_filename=job_name,
                 use_threads=not args.mpi)
        wait(job_name)

        results.append(collect_results(job_name))

    save_results(args.results, results)
