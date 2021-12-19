# OpenMP parallelized determinant computation

![](openmp/graphs/polus/heatmap.png)
![](openmp/graphs/polus/lineplot.png)
![](openmp/graphs/polus/lineplot_small.png)
![](openmp/graphs/polus/lineplot_big.png)

# MPI parallelized determinant computation

![](mpi/graphs/polus/heatmap.png)
![](mpi/graphs/polus/lineplot.png)
![](mpi/graphs/polus/lineplot_small.png)
![](mpi/graphs/polus/lineplot_big.png)

# Conclusion
Process parallelization comes with a lot of process communincation related computation losses so it is not reasonable to use it on smaller matrices. On larger matrices though it achieves better level of parallelization.
