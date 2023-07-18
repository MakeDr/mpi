from mpi4py import MPI
from numpy import array
from PIL import Image
import numpy as np
import colorsys
import sys, getopt

x, y = -0.38, -0.665
w, h = 1024, 1024
scale = 0.05
maxit = 1000
filename = "img_fcfs_test3.png"

# x, y = -0.38, -0.665
# w, h = 256, 256
# scale = 1
# maxit = 10
# filename = "img_fcfs_test.png"

def color(i,maxit):
	gray = int(255*i/maxit)
	return (0, gray, 100)

def iterations_at_point(x, y, maxit):
	x0 = x
	y0 = y
	iter =0
	while (x*x+y*y<=4) and (iter<maxit):
		xt=x*x-y*y+x0
		yt=2*x*y + y0
		x=xt
		y=yt
		iter+=1
	return iter

# do not modify from this line
def main(argv, rank):
	global x, y
	global w, h
	global scale
	global maxit
	global filename
	try: 
		opts, args = getopt.getopt(argv,"x:y:s:W:H:m:o:h") 
	except getopt.GetoptError: 
        if (rank==0):
            print ('mandel.py -x xcenter -y ycenter -s scale -W width -H height -m maxit -o filename')
        sys.exit()
    elif opt in ("-x"):
        x = float(arg)
    elif opt in ("-y"):
        y = float(arg)
    elif opt in ("-s"):
        scale = float(arg)
    elif opt in ("-W"):
        w = int(arg)
    elif opt in ("-H"):
        h = int(arg)
    elif opt in ("-m"):
        maxit = int(arg)
    elif opt in ("-o"):
        filename = arg
    if rank==0:
        print("mandel: x=", x, "y=", y, "scale=", scale, "width=", w, "height=", h, "maxit=", maxit, "output=", filename)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

main(sys.argv[1:], rank)
start_time = MPI.Wtime()
# do not modify to this line

# ↓ modified lines ↓
xmin = x - scale
xmax = x + scale
ymin = y - scale
ymax = y + scale



if size < 4 : 
    
    # WT와 MT의 구분 없이 혼자서 일하도록 한다.
    
    N = h // size + (w % size > rank)
    end = int(comm.scan(N))
    start = int(end - N)

    C = np.array([0]*N*w, dtype='i')
    for j in range(N):
        for i in range(w):
            x = xmin + i * (xmax-xmin)/w
            y = ymin + (j+start) * (ymax-ymin)/h
            C[j*w+i] = iterations_at_point(x, y, maxit)

    sendbuf = np.array(C)
    sendcounts = comm.gather(len(sendbuf), root=0)

    recvbuf = None
    if rank == 0:
        recvbuf = np.empty(sum(sendcounts), dtype=int)

    comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=0)
    
    if rank == 0:
        C = recvbuf
        image = Image.new('RGB', (w,h))
        pixels = image.load()
        for j in range(h):
            for i in range(w):
                pixels[i, j] = color(C[j*w+i], maxit)
        #print("Gathered array: {}".format(recvbuf))

        #print(C.sum())
        image.save(filename)
    
else :
    layor_size = h
    N = 1
    sended = [0] * size
    
    if rank == 0:
        # Managing Thread : Working Thread와 통신하며 Working Thread가 일할 Level를 전달한다.
        level = 0
        
        while level < layor_size :
            
            worker = int( comm.recv( ) )
            comm.send( level, dest = worker )
    
            sended[ worker ] += w + 1  # w = 가로 길이, w + 1 인 이유는 각각 앞에 layor를 추가하기 위해서
            level += 1
            
        # 만약 level이 layor_size와 같아지면, 더 이상 계산할 이유가 없다.
        # Working Terminatition Code인 -1을 각각의 Worker Thread에게 전송하여 계산을 끝마친다.
            
        for i in range( size - 1 ) :
            worker = int( comm.recv( ) )
            comm.send( -1 , dest = worker )
            comm.send( sended, dest = worker )
            
        # Gatherv를 사용하여 데이터를 1차원 array로 받는다.
        
        recv_image = np.empty( sum( sended ), dtype = int )
        
        comm.Gatherv( sendbuf = np.array([]), recvbuf = ( recv_image, sended ) )
        
        
        C = np.array([0]*h*w, dtype='i')
        
        #print(C,C.shape)
        #print(recv_image,recv_image.shape)
        
        i = 0
        while i < len(recv_image):
            C[ (h - recv_image[i] - 1) * w : (h - recv_image[i] - 1) * w + w ] = recv_image[ i + 1 : i + 1 + w ]
            i += w + 1
        
        # image = Image.new('RGB', (w,h))
        # pixels = image.load()
        # # for j in range(h):
        # #     for i in range(w):
        # #         pixels[i, j] = color(C[j*w+i], maxit)
        # print(pixels)
        # # print(C.sum(), C.shape)
        # image.save(filename)
        
        pixels = np.array(list(map(lambda i : color(i, maxit) , C))).flatten().reshape(w,h,3).astype(np.uint8)
        # print(pixels)
        image = Image.fromarray(pixels)
        image.save(filename)

    else :
        
        # Worker Thread : Managing Thread에게 일할 Level을 전송받고 그 데이터를 전송해준다.
        
        storage = []
        
        while True :
            comm.send( rank, dest = 0 )
            working_level = comm.recv( )
        
            if working_level == -1 :
                # print('end')
                sended = comm.recv( )
                break
            
            start = h - working_level - 1
            
            # Working Level에 맞는 Layor에 대해 작업을 수행한다.
            C = [0]*N*w
            for j in range(N):
                for i in range(w):
                    x = xmin + i * (xmax-xmin)/w
                    y = ymin + (j+start) * (ymax-ymin)/h
                    C[j*w+i] = iterations_at_point(x, y, maxit)
            
            # Working Level을 특정지을 수 있게 잘 저장한다.
            
            storage.append( [working_level] + C )
        
        # 이제 Managing Thread에게 데이터를 전송해야한다.
        
        # 'storage'라는 잘 가공된 array가 있다고 하면, 코드는 다음과 같다 :
        sendbuf = np.array(storage, dtype='i')
        recvbuf = None
        
        comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sended), root=0)

# do not modify from this line
end_time = MPI.Wtime()
if rank == 0:
    print("\nNormal : Overall elapsed time: " + str(end_time-start_time))