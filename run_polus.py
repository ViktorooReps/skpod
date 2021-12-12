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

    if args.mpi:
        compile_args = ['-qsmp=omp', '-O5', '-o', args.exec_file]
        mpixlc_compile(args.src_file, compile_args)
    # else:
    #     compile_args = ['-qsmp=omp', '-o', args.exec_file]
    #     xlc_compile(args.src_file, compile_args)
    else:
        compile_args = ['-qsmp=omp', '-o', '-O5', args.exec_file]
        mpixlc_compile(args.src_file, compile_args)

    if not os.path.exists(args.exec_file):
        raise ValueError('Compilation failed!')

    results = []
    for n_proc in args.processes:
        job_name = args.experiment + '-' + str(n_proc) + ("p" if args.mpi else "t")

        schedule(Machine.POLUS, n_proc, exec_file=args.exec_file, res_filename=job_name,
                 use_threads=not args.mpi)
        wait(job_name)

        results.append(collect_results(job_name))

    save_results(args.results, results)

