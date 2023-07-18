from mpi4py import MPI
import numpy as np
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

local = random.randint(2, 5)
print("rank : {0}, local : {1}".format(rank, local))

sum = comm.reduce(local, MPI.SUM, root=0)
if (rank==0):
    print("sum : {}".fotmat(sum))