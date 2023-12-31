from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

recvbuf = comm.gather(rank)
if (rank == 0):
    print("Gathered array : {}".format(recvbuf))