from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

local_array = [rank] * 4
print("rank : {0}, local_array : {1}".format(rank, local_array))
sendbuf = np.array(local_array)

recvbuf = None
if (rank == 0):
    recvbuf = np.empty(size*4, dtype=int)

comm.Gather(sendbuf=sendbuf, recvbuf=recvbuf)
if (rank == 0):
    print("Gathered array : {}".format(recvbuf))