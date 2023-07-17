from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if(rank == 0):
    msg = "Hell\'o MPI"
    comm.send(msg, dest=1)
elif(rank == 1):
    s = comm.recv()
    print("rank_{0} : {1}".format(rank, s))
