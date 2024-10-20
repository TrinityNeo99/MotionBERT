"""
@Project: 2024-human-pose-estimation-tutorial
@FileName: find_gpus.py
@Description: 自动描述，请及时修改
@Author: Wei Jiangning
@version: 1.0.0a1.0
@Date: 2024/10/3 17:54 at PyCharm
"""
import os
import sys
import time
import platform

cmd = 'python my_train.py --config configs/pose3d/MB_train_binocular_pingpong.yaml --checkpoint checkpoint/pose3d/binocular_pingpong'

def gpu_info():
    system = platform.system()
    if system == "Windows":
        gpu_status = os.popen('nvidia-smi | findstr %').read().split('|')
    else:  # if system == "Linux"
        gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
    gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
    return gpu_power, gpu_memory


def narrow_setup(interval=2):
    gpu_power, gpu_memory = gpu_info()
    i = 0
    start = time.time()
    while gpu_memory > 1000 or gpu_power > 20:  # set waiting condition
        gpu_power, gpu_memory = gpu_info()
        i = i % 5
        symbol = 'monitoring ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_power_str = 'gpu power %d W |' % gpu_power
        gpu_memory_str = 'gpu memory %d MiB |' % gpu_memory
        sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
        time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        sys.stdout.write('\n' + f"waiting for {time_str}")
        sys.stdout.flush()
        time.sleep(interval)
        i += 1
    print('n' + cmd)
    print(time.localtime())
    os.system(cmd)


if __name__ == '__main__':
    narrow_setup()
