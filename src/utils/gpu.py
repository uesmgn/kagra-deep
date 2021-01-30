import subprocess
import time


def getTemp():
    cmd = "nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"
    maxtemp = max([int(tmp) for tmp in subprocess.check_output(cmd).split()])
    return maxtemp


def coolGPU(lower=50, upper=70):
    temp = getTemp()
    print("GPU temperature: %d C" % temp)
    if temp > upper:
        print("cooling GPU...")
        while temp > lower:
            temp = getTemp()
            print("GPU temperature: %d C" % temp)
            time.sleep(10)
        print("GPU temperature: %d C" % temp)
