import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import dataframe_image as dfi

if __name__ == '__main__':

    # creating graphs for openmp from polus

    data_openmp = pd.read_csv('openmp/data/polus/data.csv', sep='\t')
    data_openmp = data_openmp[data_openmp['n'] > 2]
    small_data = data_openmp[data_openmp['n'] < 2 ** 6]
    big_data = data_openmp[data_openmp['n'] > 2 ** 7]

    palette = sns.color_palette("flare", 7)
    sns.set()

    ax = sns.lineplot(data=data_openmp, x='n', y='time', hue='threads', palette=palette)
    ax.set_title('Execution time in seconds with nxn matrix')
    ax.set_xscale('log', basex=2)
    plt.savefig('openmp/graphs/polus/lineplot.png')
    plt.show()

    ax = sns.lineplot(data=small_data, x='n', y='time', hue='threads', palette=palette)
    ax.set_title('Execution time in seconds with nxn matrix on smaller n')
    ax.set_xscale('log', basex=2)
    plt.savefig('openmp/graphs/polus/lineplot_small.png')
    plt.show()

    ax = sns.lineplot(data=big_data, x='n', y='time', hue='threads', palette=palette)
    ax.set_title('Execution time in seconds with nxn matrix on larger n')
    ax.set_xscale('log', basex=2)
    plt.savefig('openmp/graphs/polus/lineplot_big.png')
    plt.show()

    data_openmp = data_openmp.pivot("n", "threads", "time")
    dfi.export(data_openmp, 'openmp/graphs/polus/table.png')

    ax = sns.heatmap(data_openmp, norm=LogNorm(), square=True)
    ax.set_title('Execution time in seconds with nxn matrix')
    plt.savefig('openmp/graphs/polus/heatmap.png')
    plt.show()

    # creating graphs for mpi from polus

    data_mpi = pd.read_csv('mpi/data/polus/data.csv', sep='\t')
    data_mpi = data_mpi[data_mpi['n'] > 2]
    small_data = data_mpi[data_mpi['n'] < 2 ** 6]
    big_data = data_mpi[data_mpi['n'] > 2 ** 7]

    palette = sns.color_palette("flare", 7)
    sns.set()

    ax = sns.lineplot(data=data_mpi, x='n', y='time', hue='processes', palette=palette)
    ax.set_title('Execution time in seconds with nxn matrix')
    ax.set_xscale('log', basex=2)
    plt.savefig('mpi/graphs/polus/lineplot.png')
    plt.show()

    ax = sns.lineplot(data=small_data, x='n', y='time', hue='processes', palette=palette)
    ax.set_title('Execution time in seconds with nxn matrix on smaller n')
    ax.set_xscale('log', basex=2)
    plt.savefig('mpi/graphs/polus/lineplot_small.png')
    plt.show()

    ax = sns.lineplot(data=big_data, x='n', y='time', hue='processes', palette=palette)
    ax.set_title('Execution time in seconds with nxn matrix on larger n')
    ax.set_xscale('log', basex=2)
    plt.savefig('mpi/graphs/polus/lineplot_big.png')
    plt.show()

    data_mpi = data_mpi.pivot("n", "processes", "time")
    dfi.export(data_mpi, 'mpi/graphs/polus/table.png')

    ax = sns.heatmap(data_mpi, norm=LogNorm(), square=True)
    ax.set_title('Execution time in seconds with nxn matrix')
    plt.savefig('mpi/graphs/polus/heatmap.png')
    plt.show()

    # creating graphs for mpi from bluegene

    data_mpi = pd.read_csv('mpi/data/bluegene/data.csv', sep='\t')
    data_mpi = data_mpi[data_mpi['n'] > 2]
    small_data = data_mpi[data_mpi['n'] < 2 ** 6]
    big_data = data_mpi[data_mpi['n'] > 2 ** 7]

    palette = sns.color_palette("flare", 8)
    sns.set()

    ax = sns.lineplot(data=data_mpi, x='n', y='time', hue='processes', palette=palette)
    ax.set_title('Execution time in seconds with nxn matrix')
    ax.set_xscale('log', basex=2)
    plt.savefig('mpi/graphs/bluegene/lineplot.png')
    plt.show()

    ax = sns.lineplot(data=small_data, x='n', y='time', hue='processes', palette=palette)
    ax.set_title('Execution time in seconds with nxn matrix on smaller n')
    ax.set_xscale('log', basex=2)
    plt.savefig('mpi/graphs/bluegene/lineplot_small.png')
    plt.show()

    ax = sns.lineplot(data=big_data, x='n', y='time', hue='processes', palette=palette)
    ax.set_title('Execution time in seconds with nxn matrix on larger n')
    ax.set_xscale('log', basex=2)
    plt.savefig('mpi/graphs/bluegene/lineplot_big.png')
    plt.show()

    data_mpi = data_mpi.pivot("n", "processes", "time")
    dfi.export(data_mpi, 'mpi/graphs/bluegene/table.png')

    ax = sns.heatmap(data_mpi, norm=LogNorm(), square=True)
    ax.set_title('Execution time in seconds with nxn matrix')
    plt.savefig('mpi/graphs/bluegene/heatmap.png')
    plt.show()
