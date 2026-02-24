# Servo
Servo for SpIOMM

# Real time kernel

## install RT kernel
- Use Ubuntu 24.04 LTS
- activate pro version to get real-time (PREEMPT_RT) kernel
- In the BIOS disable CPU optimization (multithreading or SMT, turbo/boost, P-states, CPPC), energy saving (global C-states, Power Supply Idel Control),  Virtualization (IOMMU), Spread Spectrum and ASPM (PCIe)

## test RT
```
sudo cyclictest -p95 -t1 -i100
```
should show a max jitter smaller than 30 us

```
sudo stress-ng --cpu 4 --io 2 --vm 2 --vm-bytes 1G --timeout 30s &  sudo cyclictest -p95 -t1 -i100
```
should show a larger jitter but still smaller than 30 us

# Install drivers and software

## Install python environment

### Install Miniconda

https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2

### Install servo env

download the [installation file](./servo.yml) then run:

```
conda env create -f servo.yml
```

## Install software
```
git clone https://github.com/thomasorb/Servo.git
conda activate servo
cd Servo
python servo.py install # for simple user but not for developper
```

### developper only

Add module
```
# add to file: ~/miniconda3/envs/servo/lib/python3.12/site-packages/conda.pth
/path/to/Servo
```

Add command
```
ln -s /absolute/path/to/Servo/bin/servo ~/miniconda3/envs/servo/
```




### Install 

