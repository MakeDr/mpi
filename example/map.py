from mpi4py import MPI
import numpy as np
import math

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

x = range(20)
m = int(math.ceil(float(len(x))/size))
x_chunk = x[rank * m : (rank+1) * m]
r_chunk = map(math.sqrt, x_chunk)
r = comm.reduce(list(r_chunk))
if (rank == 0):
    print(r)
#serial => print(list(map(math.sqrt, x))) <일반 코드>