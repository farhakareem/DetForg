import multiprocessing
import cv2
import numpy as np
import os
import subprocess as sp
import math
import sys




def temp(yuv_filename,width,height):
    DF=list()
    OF=list()
    DF_=list()
    OF_=list()

    type=int()



    def type1(yuv_filename, width, height, q):
        #print("In Type 1")
        DF=set()
        OF=set()
        t1=int()

        def corr(s,t):
            if s==t:
                return 0
            else:
                if s in S:
                    c=S_map[s]
                    return cor[c][t]
                else:
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
        n_frames = file_size // (width*height*3 // 2)
        f = open(yuv_filename, 'rb')
        old_yuv = np.frombuffer(f.read(width*height*3//2), dtype=np.uint8).reshape((height*3//2, width))
        old_gray = cv2.cvtColor(old_yuv, cv2.COLOR_YUV2GRAY_I420)

        for i in range(1,n_frames):
            yuv = np.frombuffer(f.read(width*height*3//2), dtype=np.uint8).reshape((height*3//2, width))
            gray = cv2.cvtColor(yuv, cv2.COLOR_YUV2GRAY_I420)
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

        S=[]
        L=[]
        N=len(sum_of)
        THR_F=1.5

        for i in range(2, N-2):
            avgsum_of = 0.25*(sum_of[i-1]+sum_of[i+1]+sum_of[i-2]+sum_of[i+2])
            beta = sum_of[i]/avgsum_of
            if beta > THR_F:
                S.append(i-1)
                S.append(i)
                S.append(i+1)

        #print('S: ', S)

        D1=[]
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
            
        S_map = dict(map(lambda t: (t[1], t[0]), enumerate(S)))
        cor = np.empty([len(S),N], dtype=np.float64)
        for i in S:
            c=S_map[i]
            for j in range(N):
                if i==j:
                    cor[c][j] = 0
                    continue
                nrsum=dr1sum=dr2sum=0
                for m in range(h):
                    for n in range(w):
                        nr1=(of[i][m][n]-avgof[i])
                        nr2=(of[j][m][n]-avgof[j])
                        nr=nr1*nr2
                        nrsum+=nr
                        dr1=nr1*nr1
                        dr1sum+=dr1
                        dr2=nr2*nr2
                        dr2sum+=dr2
                res=abs(nrsum/math.sqrt(dr1sum*dr2sum))
                cor[c][j]=res
            max_cor=np.amax(cor[c])
            index=np.argmax(cor[c])
            avgcor=np.mean(cor[c])
            if avgcor<0.2:
                THR_C1=0.3
            else: 
                THR_C1=avgcor*2
            if max_cor>THR_C1:
                D1.append([i,index])
                D1.append([i+1,index+1])

        #print('D: ', D1)
        THR_C2=0.2
        W=9
        for z in reversed(D1):
            if abs(z[1]-z[0]) < W:
                D1.remove(z)
            else: 
                prev_cor=corr(z[0]-1,z[1]-1)
                next_cor=corr(z[0]+1,z[1]+1)
                if prev_cor < THR_C2 and next_cor < THR_C2:
                    D1.remove(z)
                elif prev_cor > THR_C2 and next_cor > THR_C2:
                    if prev_cor!=1.0:
                        if next_cor==1.0:
                            continue          
                        else:
                            D1.remove(z)         
                    elif next_cor!=1.0:
                        continue
                    else:
                        D1.remove(z)

        

        for i in range(len(D1)):
            for j in range(len(D1)):
                if i!=j:
                    if D1[i][0]<D1[j][0] and D1[i][1]<D1[j][1] and abs(D1[i][0]-D1[i][1])==abs(D1[j][0]-D1[j][1]) and abs(D1[i][0]-D1[i][1])>1 and abs(D1[i][0]-D1[j][0])==abs(D1[i][1]-D1[j][1]) and abs(D1[i][0]-D1[j][0])>8:
                        for a in range(D1[i][0], D1[j][0]+1):
                            DF.add(a)
                        for b in range(D1[i][1], D1[j][1]+1):
                            OF.add(b)

        DF=DF.difference(DF.intersection(OF))
        DF=list(DF)
        DF.sort()
        OF=list(OF)
        OF.sort()
        '''
        print('D after false detection reduction: ', D1)
        #print("Type 1 Forgery - Simple Cloning Copy-Move: ")
        print("Duplicated Frames: ", end=" ")
        print(DF)   
        print("Original Frames: ", end=" ")
        print(OF)
        '''
        if DF:
            f = open(yuv_filename, 'rb')
            o = open('c:/Users/PC/projects/detforg/static/Original.yuv', 'wb')
            d = open('c:/Users/PC/projects/detforg/static/Duplicated.yuv', 'wb')
            fr_count=0
            fr_size=int(file_size/n_frames)
            while fr_count<n_frames:
                frame=f.read(fr_size)
                if fr_count in DF:
                    d.write(frame)
                else:
                    o.write(frame)
                fr_count+=1
            d.close()
            o.close()
            f.close()
            sp.run('c:/Users/PC/projects/detforg/calc/ffmpeg -loglevel quiet -y -s {}x{} -pixel_format yuv420p -i c:/Users/PC/projects/detforg/static/Original.yuv -vcodec libx264 -crf 17 -pix_fmt yuv420p c:/Users/PC/projects/detforg/static/Original.mp4'.format(width, height))
            sp.run('c:/Users/PC/projects/detforg/calc/ffmpeg -loglevel quiet -y -s {}x{} -pixel_format yuv420p -i c:/Users/PC/projects/detforg/static/Duplicated.yuv -vcodec libx264 -crf 17 -pix_fmt yuv420p c:/Users/PC/projects/detforg/static/Duplicated.mp4'.format(width, height))
            os.remove('c:/Users/PC/projects/detforg/static/Original.yuv')
            os.remove('c:/Users/PC/projects/detforg/static/Duplicated.yuv')
            type=1
            q.put(type)
            q.put(DF)
            q.put(OF)
















    def type2(yuv_filename, width, height, q):
        #print("In Type 2")
        DF_=set()
        OF_=set()
        t2=int()
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
        n_frames = file_size // (width*height*3 // 2)
        f = open(yuv_filename, 'rb')

        old_yuv = np.frombuffer(f.read(width*height*3//2), dtype=np.uint8).reshape((height*3//2, width))
        old_gray = cv2.cvtColor(old_yuv, cv2.COLOR_YUV2GRAY_I420)

        for i in range(1,n_frames):
            yuv = np.frombuffer(f.read(width*height*3//2), dtype=np.uint8).reshape((height*3//2, width))
            gray = cv2.cvtColor(yuv, cv2.COLOR_YUV2GRAY_I420)
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
        #print("Type 2 Forgery - Smooth Insertion Copy-Move: ")
        print("Duplicated Frames: ", end=" ")
        print(DF_)
        print("Original Frames: ", end=" ")
        print(OF_)
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
            sp.run('c:/Users/PC/projects/detforg/calc/ffmpeg -loglevel quiet -y -s {}x{} -pixel_format yuv420p -i c:/Users/PC/projects/detforg/static/Original_.yuv -vcodec libx264 -crf 17 -pix_fmt yuv420p c:/Users/PC/projects/detforg/static/Original_.mp4'.format(width, height))
            sp.run('c:/Users/PC/projects/detforg/calc/ffmpeg -loglevel quiet -y -s {}x{} -pixel_format yuv420p -i c:/Users/PC/projects/detforg/static/Duplicated_.yuv -vcodec libx264 -crf 17 -pix_fmt yuv420p c:/Users/PC/projects/detforg/static/Duplicated_.mp4'.format(width, height))
            os.remove('c:/Users/PC/projects/detforg/static/Original_.yuv')
            os.remove('c:/Users/PC/projects/detforg/static/Duplicated_.yuv')
            type=2
            q.put(type)
            q.put(DF_)
            q.put(OF_)










    if __name__=='temp':
        q = multiprocessing.Queue()
        p1=multiprocessing.Process(target=type1, args=(yuv_filename, width, height, q))
        p1.start()
        p2=multiprocessing.Process(target=type2, args=(yuv_filename, width, height, q))
        p2.start()
        type=q.get()
        if type==1:
            msg="Type 1 Forgery - Simple Cloning Copy-Move"
            print(msg)
            DF=q.get()
            print(DF)
            OF=q.get()
            print(OF)
            return(msg)
            #p2.terminate()
        elif type==2:
            msg="Type 2 Forgery - Smooth Insertion Copy-Move"
            print(msg)
            DF_=q.get()
            print(DF_)
            OF_=q.get()
            print(OF_)
            return(msg)
            #p1.terminate()