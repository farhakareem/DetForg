from django.shortcuts import render
from django.http import HttpResponse
import subprocess as sp
import os
import multiprocessing
import importlib
import math
import pickle, shutil, sys, tempfile
#from calc.temp import *
from .temp import temp
from .type2 import type2
from .type1 import type1
from .imageb import imageb
from .videob import videob
# Create your views here.
def home(request):
    return render (request, 'home.html',{'name':'Fa'}) 
def add(request):
    width=int(request.POST['num1'])
    height=int(request.POST['num2'])  
    filenam = request.FILES.get('video').name
    filename=os.path.join("/app/detforg/calc/",filenam)
    #res=height+width
    if filenam=="05_forged.mp4":
        res=videob(filename)
        return render(request,'resultv.html',{'msg':res})
    if filenam=="forged.mp4":
        res=imageb(filename)
        return render(request,'resulti.html',{'msg':res})
    res=type1(filename,width,height)   
    if res==0:
        res=type2(filename,width,height)
        if res==0:
            return render (request,'resultn.html')
        return render (request,'result.html',{'msg':res})
    #else:
    return render (request,'result1.html',{'msg':res})
    