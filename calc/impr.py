from temp import *
filen="can_0.yuv"
wi=320
he=240
if __name__=='__main__':
        q = multiprocessing.Queue()
        p1=multiprocessing.Process(target=type1, args=(yuv_filename, width, height, q))
        p1.start()
        p2=multiprocessing.Process(target=type2, args=(yuv_filename, width, height, q))
        p2.start()
        print("ho")
        type=q.get()
        if type==1:
            msg="Type 1 Forgery - Simple Cloning Copy-Move"
            return(msg)
            print(msg)
            DF=q.get()
            print(DF)
            OF=q.get()
            print(OF)
            #p2.terminate()
        elif type==2:
            msg="Type 2 Forgery - Smooth Insertion Copy-Move"
            return(msg)
            print(msg)
            DF_=q.get()
            print(DF_)
            OF_=q.get()
            print(OF_)
            #p1.terminate()
msg=temp(filen,wi,he)
print (msg)