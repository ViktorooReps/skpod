import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import dataframe_image as dfi

if __name__ == '__main__':
    data = pd.read_csv('data.csv', sep='\t')

    palette = sns.color_palette("flare", 8)
    sns.set()
    ax = sns.lineplot(data=data, x='n', y='time', hue='threads', palette=palette)
    ax.set_title('Execution time in seconds with nxn matrix')
    ax.set_xscale('log', basex=2)
    plt.savefig('lineplot.png')
    plt.show()

    data = data.pivot("n", "threads", "time")
    dfi.export(data, 'table.png')

    ax = sns.heatmap(data, norm=LogNorm(), square=True)
    ax.set_title('Execution time in seconds with nxn matrix')
    plt.savefig('heatmap.png')
    plt.show()


