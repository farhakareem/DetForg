#importing required libraries
import cv2
import numpy as np
import os

name='forged'
def imageb(filename):
    #the video is read
    cap = cv2.VideoCapture(filename)
    #no of frames in video
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tot=total-1

    #initializing video residual matrix, binary residual mask
    r = np.zeros((240,320,tot),dtype=np.int)
    m = np.zeros((240,320,tot),dtype=np.int)
    #time frame
    t=0

    #calculating residual matrix, binary residual mask
    ret, frame = cap.read()
    #grayscaled the frame
    prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    while(1):
        
        ret, frame = cap.read()
        if ret==False: break
        
        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for i in range(240):
            for j in range(320):
                r[i][j][t]= int(prev[i][j])-int(next[i][j])

                if r[i][j][t]==0 :
                    m[i][j][t]=1
                        
        t+=1   
        prev=next

    #Morphological Erosion
    a,b,c = 12,12,10
    a1=a-1
    b1=b-1
    c1=c-1
    kernel = np.ones((a,b,c), np.int) 
    x = np.zeros((a,b,c), np.int) 
    e = np.zeros((240,320,tot),dtype=np.int)
    for i in range(240-a):
        for j in range(320-b):
            for k in range(tot-c):
                x=m[i:i+a , j:j+b , k:k+c]
                if ( (x==kernel).all() ):
                    e[i+a1][j+b1][k+c1]=1

    #initialising feature vector
    f = np.zeros((240,320,2),dtype=np.int)
    l = np.zeros((tot),dtype=np.int)

    #calculating Feature Vector Matrix
    maxi=t=0
    for i in range(240):
        for j in range(320):
            l=e[i][j]
            maxi=t=0
            for k in range(tot):
                count=0
                for g in range(k,tot):
                    if l[g]==0:
                        break
                    count=count+1
                if(count>maxi):
                    maxi=count
                    t=k
            f[i][j]=[maxi,t]

    #find possible fij1 values
    p= np.zeros((76800),dtype=np.int)
    t=0
    for i in range(240):
        for j in range(320):
            if(f[i][j][0]!=0):
                p[t]=f[i][j][0]
                t+=1
    s=set(p)
    s=sorted(s,reverse=True)

    #find highest feasible fij1 value
    for i in s:
        count=0
        for j in p:
            if(i==j):
                count+=1
        if count>=10:
            maxi=i
            break

    #finding the mask values
    flag=0
    for i in range(240):
        for j in range(320):
            if(f[i][j][0]==maxi):
                flag=1
                time=f[i][j][1]
                break
        if flag==1:
            break

    cap.release()
    cv2.destroyAllWindows()

    #masking the forged area
    #the video is read
    cap = cv2.VideoCapture('c:/Users/PC/projects/detforg/calc/forged.mp4')
    fourcc=cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter('c:/Users/PC/projects/detforg/static/detect.mp4',fourcc, 30.0 , (320,240))
    end=time+maxi-3
    for n in range(tot):
        ret, frame = cap.read()
        if(time<=n<end):
            cv2.rectangle(frame, (i,j+30) , (i+80,j+70) , color=(0, 0, 255) , thickness=2)
        video.write(frame) 
    msg="Image-based Spatio-temporal forgery"   
    video.release()
    cap.release()
    cv2.destroyAllWindows()
    return msg