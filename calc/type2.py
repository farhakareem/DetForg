import cv2
import numpy as np
import os
import subprocess as sp
import math
import sys

#yuv_filename = sys.argv[1]
#width, height = int(sys.argv[2]), int(sys.argv[3])

def type2(yuv_filename,width,height):
    def corr(s,t):
        nrsum=dr1sum=dr2sum=0
        for m in range(h):
            for n in range(w):
                nr1=(of[s][m][n]-avgof[s])
                nr2=(of[t][m][n]-avgof[t])
                nr=nr1*nr2
                nrsum+=nr
                dr1=nr1*nr1
                dr1sum+=dr1
                dr2=nr2*nr2
                dr2sum+=dr2
        res=abs(nrsum/math.sqrt(dr1sum*dr2sum))
        return res

    of=[]
    sum_of=[]

    file_size = os.path.getsize(yuv_filename)
    #calculating the number of frames
    n_frames = file_size // (width*height*3 // 2)
    f = open(yuv_filename, 'rb')
    #the first frame in 8 bit single channel format
    old_yuv = np.frombuffer(f.read(width*height*3//2), dtype=np.uint8).reshape((height*3//2, width))
    #converting the frame into gray-scale
    old_gray = cv2.cvtColor(old_yuv, cv2.COLOR_YUV2GRAY_I420)

    for i in range(1,n_frames):
        #next frame
        yuv = np.frombuffer(f.read(width*height*3//2), dtype=np.uint8).reshape((height*3//2, width))
        gray = cv2.cvtColor(yuv, cv2.COLOR_YUV2GRAY_I420)
        #Gunnar Farneback Optical Flow Calculation returns a resulting flow image into flow
        flow = cv2.calcOpticalFlowFarneback(old_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fsum = 0
        for m in range(height):
            for n in range(width):
                fsum = fsum+(abs(flow[m][n][0])+abs(flow[m][n][1]))
        sum_of.append(fsum)  
        mat = np.empty([height,width], dtype=np.float32)
        for r in range(height):
            for c in range(width):   
                mat[r][c] = flow[r][c][1]
        mat = mat[::2,::2]
        of.append(mat) 
        old_gray = gray.copy()
    f.close()

    N=len(sum_of)
    avgof=[]
    h=int(height/2)
    w=int(width/2)
    tot_res=h*w
    for i in range(N):
        tot_of=0
        for m in range(h):
            for n in range(w):
                tot_of+=of[i][m][n]
        avgof.append((tot_of/tot_res)) 

    L=[]

    for i in range(3, N-2):
        count=0
        for k in range(0, 3):
            val=sum_of[i+k]/sum_of[i-k-1]
            if (val>0.9) and (val<(1.2)):
                count=count+1
        if count==3:
            L.append(i)

    #print('L: ', L)

    D2=[]

    THR_C2=0.7
    for i in L:
        k=0
        while corr(i+k,i-k-1) >= THR_C2 and corr(i+k+1,i-k-2) >= THR_C2:
            if i-k-4==0 or i+k+3>N-1:
                break        
            else:
                k=k+2
        D2.append([i-k-1,i,i+k+1])

    #print('D: ', D2)

    W=10

    DF_=set()
    OF_=set()

    for i in reversed(D2):
        if abs(i[0]-i[2]) < 2*W-2:
            D2.remove(i)

    if len(D2)==1:
        for i in D2:
            for a in range(i[1]+1,i[2]+2):
                DF_.add(a)
            for b in range(i[0]-1,i[1]):
                OF_.add(b)
    else:
        for i in reversed(D2):
            for a in range(i[1]+1,i[2]):
                DF_.add(a)
            for b in range(i[0]+1,i[1]):
                OF_.add(b)



    OF_=OF_.difference(OF_.intersection(DF_))

    DF_=list(DF_)
    DF_.sort()

    OF_=list(OF_)
    OF_.sort()
    '''
    print('D after false detection reduction: ', D2)
    print("Type 2 Forgery - Smooth Insertion Copy-Move: ")
    print("Duplicated Frames: ", end=" ")
    print(DF_)
    print("Original Frames: ", end=" ")
    print(OF_)
    msg="Type 2 Forgery - Smooth Insertion Copy-Move: "
    '''
    if DF_:
        f = open(yuv_filename, 'rb')
        o = open('c:/Users/PC/projects/detforg/static/Original_.yuv', 'wb')
        d = open('c:/Users/PC/projects/detforg/static/Duplicated_.yuv', 'wb')
        fr_count=0
        fr_size=int(width*height*1.5)
        while fr_count<n_frames:
            frame=f.read(fr_size)
            if fr_count in DF_:
                d.write(frame)
            else:
                o.write(frame)
            fr_count+=1
        d.close()
        o.close()
        f.close()
        sp.run('c:/Users/PC/projects/detforg/calc/ffmpeg -y -s {}x{} -pixel_format yuv420p -i c:/Users/PC/projects/detforg/static/Original_.yuv -vcodec libx264 -crf 17 -pix_fmt yuv420p c:/Users/PC/projects/detforg/static/Original_.mp4'.format(width, height))
        sp.run('c:/Users/PC/projects/detforg/calc/ffmpeg -y -s {}x{} -pixel_format yuv420p -i c:/Users/PC/projects/detforg/static/Duplicated_.yuv -vcodec libx264 -crf 17 -pix_fmt yuv420p c:/Users/PC/projects/detforg/static/Duplicated_.mp4'.format(width, height))
        os.remove('c:/Users/PC/projects/detforg/static/Original_.yuv')
        os.remove('c:/Users/PC/projects/detforg/static/Duplicated_.yuv')
        msg="Type 2 Forgery - Smooth Insertion Copy-Move"
        print(msg)
        print(DF_)
        print(OF_)
    else:
        msg=0
    return msg





