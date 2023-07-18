from mpi4py import MPI
import numpy as np
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
local_array = [rank] * random.randint(2, 5)
print("rank : {0}, local_array : {1}".format(rank, local_array))

sendbuf = np.array(local_array)
counts = comm.gather(len(sendbuf), root=0)
recvbuf = None

if (rank == 0):
    print("counts : {0}, total : {1}".format(counts, sum(sendbuf,counts)))
    recvbuf = np.empty(sum(counts), dtype=int)
comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, counts),root=0)
if (rank == 0):
    print("Gathered array : {}".format(recvbuf))