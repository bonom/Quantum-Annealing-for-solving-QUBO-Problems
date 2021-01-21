from time import sleep
from datetime import timedelta
import sys

def write(dir, string):
    file = open(dir, 'a')
    file.write(string+'\n')
    file.close()

pass_time = 0
while True:
    try:
        sleep(60)
        pass_time += 1
        tm = timedelta(seconds = pass_time)
        string = "Tempo: " + str(tm) 
        print(string)
        write('tm.txt',string)
    except:
        string = "Terminato per colpa di " + str(sys.exc_info()[0])
        print(string)
        write('tm.txt',string)
        exit()
