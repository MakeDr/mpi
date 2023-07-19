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
		sys.exit(2) 
	for opt, arg in opts: 
		if opt == '-h': 
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
            print ("mandel: x=", x, "y=", y, "scale=", scale, "width=", w, "height=", h, "maxit=", maxit, "output=", filename)

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

if size < 3 :
    # 예시 주석: 농장이 하나일 경우(농가가 독자적으로 일을 함)
    if rank == 0:
        # 전문가 주석: 프로세스 0이 관리자 프로세스와 작업자 프로세스의 역할을 겸한다.
        
        # 코드 뭉치 1
        # 전문가 주석: 각 그리드 포인트에서 수행할 반복 횟수 계산
        # 작동 방식: xmin, ymin, xmax, ymax를 기준으로 그리드에 해당하는 각 좌표별로
        #            iterations_at_point 함수를 사용하여 반복 횟수를 계산하고 C 배열에 저장합니다.
        C = np.array([0] * h * w, dtype='i')
        for j in range(h):
            for i in range(w):
                x = xmin + i * (xmax - xmin) / w
                y = ymin + j * (ymax - ymin) / h
                C[j * w + i] = iterations_at_point(x, y, maxit)

        # 코드 뭉치 2
        # 전문가 주석: 결과 배열 C를 사용하여 이미지를 생성한 다음 파일에 저장
        # 작동 방식: 저장한 반복 횟수에 따른 색상 값을 pixels에 저장하고,
        #            이 값을 이용하여 이미지를 생성한 후 파일에 저장합니다.
        pixels = np.array(list(map(lambda i: color(i, maxit), C))).flatten().reshape(w, h, 3).astype(np.uint8)
        image = Image.fromarray(pixels)
        image.save(filename)

else:
    # 예시 주석: 농장을 나눌 경우(농부들이 존재하므로)
    
    # 전문가 주석: 작업자들간의 작업 분배 및 결과 수집을 위해 프로세스 통신 사용.
    layor_size = h
    N = 1
    sended = [0] * size
    completed_tasks = [0] * size
    
    if rank == 0:
        # 농장 주인
        
        # 코드 뭉치 3
        # 전문가 주석: 작업을 작업자들에게 분배
        # 작동 방식: 각 작업자들에게 작업 메시지를 전달하고, 레벨을 기준으로 작업을 할당합니다.
        #            각 작업자가 완료한 작업의 수에 따라 할당된 작업을 조절합니다.
        level = 0
        while level < layor_size:
            worker = int(comm.recv())
            if sum(completed_tasks) > 0:
                tasks_to_assign = max(1, int(completed_tasks[worker] / sum(completed_tasks) * len(completed_tasks)))
            else:
                tasks_to_assign = 1

            for _ in range(tasks_to_assign):
                comm.send(level, dest=worker)
                sended[worker] += w + 1
                level += 1
                
                if level >= layor_size:
                    break

        # 코드 뭉치 4
        # 전문가 주석: 작업이 완료된 작업자들에게 종료 메시지 전송
        # 작동 방식: 각 작업자에게 -1 값을 전송하여 종료 메시지를 알리고,
        #            각 작업자로부터 결과를 전달 받으려고 sended 배열을 전송합니다.
        for i in range(size - 1):
            worker = int(comm.recv())
            comm.send(-1, dest=worker)
            comm.send(sended, dest=worker)

        # 코드 뭉치 5
        # 전문가 주석: 각 작업자로부터 결과를 수집하고 이미지를 생성한 다음 파일에 저장
        # 작동 방식: MPI의 Gatherv 함수를 사용하여 각 작업자로부터 결과를 수집하고,
        #            이미지를 생성한 후 파일에 저장합니다.
        recv_image = np.empty(sum(sended), dtype=int)
        comm.Gatherv(sendbuf=np.array([]), recvbuf=(recv_image, sended))
        C = np.array([0] * h * w, dtype='i')
        i = 0
        while i < len(recv_image):
            C[(h - recv_image[i] - 1) * w: (h - recv_image[i] - 1) * w + w] = recv_image[i + 1: i + 1 + w]
            i += w + 1
        pixels = np.array(list(map(lambda i: color(i, maxit), C))).flatten().reshape(w, h, 3).astype(np.uint8)
        image = Image.fromarray(pixels)
        image.save(filename)

    else:
        # 농부들
        
        # 전문가 주석: 작업자 프로세스는 할당된 작업을 수행하고 결과를 저장한다.
        
        # 코드 뭉치 6
        # 전문가 주석: 작업자들은 중앙 관리자(rank 0)로부터 작업을 받아서 실행
        # 작동 방식: 각 작업자는 관리자로부터 할당 받은 작업을 수행한 후
        #            결과를 저장합니다. 이 저장 공간을 storage 변수로 확인합니다.
        storage = []
        while True:
            comm.send(rank, dest=0)
            working_level = comm.recv()
            if working_level == -1:
                sended = comm.recv()
                break
            start = h - working_level - 1
            C = [0] * N * w
            for j in range(N):
                for i in range(w):
                    x = xmin + i * (xmax - xmin) / w
                    y = ymin + (j + start) * (ymax - ymin) / h
                    C[j * w + i] = iterations_at_point(x, y, maxit)
            storage.append([working_level] + C)

        # 코드 뭉치 7
        # 전문가 주석: 작업자들의 결과를 관리자(rank 0)로 전송
        # 작동 방식: 각 작업자는 작업이 끝난 후 MPI의 Gatherv 함수를 사용하여
        #            결과 배열을 관리자로 전송합니다.
        sendbuf = np.array(storage, dtype='i')
        recvbuf = None
        comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sended), root=0)



# do not modify from this line
end_time = MPI.Wtime()
if rank == 0:
    print("\nFixed : Overall elapsed time: " + str(end_time-start_time))