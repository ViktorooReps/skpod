from pathlib import Path
from typing import List

from utils import create_parser, xlc_compile, schedule, Machine, wait, collect_results, save_results

if __name__ == '__main__':
    arg_parser = create_parser()
    args = arg_parser.parse_args()

    compile_args = ['-qsmp=omp', '-o', args.exec_file]
    xlc_compile(Path(args.src_file), compile_args)

    results: List[str] = []
    for n_proc in args.processes:
        job_name = args.experiment + f'-{n_proc}p'

        schedule(Machine.POLUS, n_proc, exec_file=Path(args.exec_file), res_filename=job_name)
        wait(job_name)

        results.append(collect_results(job_name))

    save_results(Path(args.results), results)

