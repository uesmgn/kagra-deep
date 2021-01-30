import subprocess
import time


def getTemp():
    cmd = "nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"
    temperature = int(subprocess.check_output(cmd))
    return temperature


def coolGPU(lower=50, upper=70):
    temp = getTemp()
    if temp > upper:
        print("GPU temperature: %d C" % temp)
        print("cooling GPU...")
        while temp > lower:
            temp = getTemp()
            print("GPU temperature: %d C" % temp)
            time.sleep(10)
        print("GPU temperature: %d C" % temp)
