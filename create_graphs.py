import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

if __name__ == '__main__':
    data = pd.read_csv('data.csv', sep='\t')
    data = data.pivot("n", "threads", "time")
    ax = sns.heatmap(data, norm=LogNorm(), square=True)
    ax.set_title('Execution time in seconds with nxn matrix')

    plt.savefig('heatmap.png')
