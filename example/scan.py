from mpi4py import MPI
import numpy as np
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

local = random.randint(2, 5)
print("rank : {0}, local : {1}".format(rank, local))

scan = comm.scan(local, MPI.SUM)
print("rank : {0}, sum : {1}".format(rank,scan))