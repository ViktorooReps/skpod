import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import dataframe_image as dfi

if __name__ == '__main__':
    data = pd.read_csv('data.csv', sep='\t')
    data = data[data['n'] > 2]
    small_data = data[data['n'] < 2 ** 6]
    big_data = data[data['n'] > 2 ** 7]

    palette = sns.color_palette("flare", 8)
    sns.set()

    ax = sns.lineplot(data=data, x='n', y='time', hue='threads', palette=palette)
    ax.set_title('Execution time in seconds with nxn matrix')
    ax.set_xscale('log', basex=2)
    plt.savefig('lineplot.png')
    plt.show()

    ax = sns.lineplot(data=small_data, x='n', y='time', hue='threads', palette=palette)
    ax.set_title('Execution time in seconds with nxn matrix on smaller n')
    ax.set_xscale('log', basex=2)
    plt.savefig('lineplot_small.png')
    plt.show()

    ax = sns.lineplot(data=big_data, x='n', y='time', hue='threads', palette=palette)
    ax.set_title('Execution time in seconds with nxn matrix on larger n')
    ax.set_xscale('log', basex=2)
    plt.savefig('lineplot_big.png')
    plt.show()

    data = data.pivot("n", "threads", "time")
    dfi.export(data, 'table.png')

    ax = sns.heatmap(data, norm=LogNorm(), square=True)
    ax.set_title('Execution time in seconds with nxn matrix')
    plt.savefig('heatmap.png')
    plt.show()


