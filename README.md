# Servo
Servo for SpIOMM

# Real time kernel

## install RT kernel
- Use Ubuntu 24.04 LTS
- activate pro version to get real-time (PREEMPT_RT) kernel
- **NVIDIA issue** When using NVIDIA graphic card, the module might not load properly. Official NVIDIA drivers must be recompiled following the instructions at https://gist.github.com/pantor/9786c41c03a97bca7a52aa0a72fa9387
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

# Install software

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

# Install drivers

## NIT IR Camera drivers

Extract file `NITLibrary-382_py312_ubu2404_amd64_bundle.tar.gz` then
```
sudo apt install -f ./NITLibrary_3.8.2_ubu2404_x86_64.deb
sudo apt install libboost-numpy1.83-dev
```
and copy the file `NITLibrary_x64_382_py312.so` found in the extracted folder (`NITLibrary-382_py312_ubu2404_amd64_bundle/NITLibrary-Python3.8.2/NITSampleCodes/`) to `Servo/servo/`

Grant more filesystem memory to USB devices by changing the file `/etc/default/grub`
```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash usbcore.usbfs_memory_mb=1000"
```
Then
```
sudo update-grub
sudo modprobe usbcore usbfs_memory_mb=1000
# sudo reboot
```

## MCC DAQ drivers

From instructions at https://github.com/mccdaq/uldaq 
```
sudo apt-get install gcc g++ make
sudo apt-get install libusb-1.0-0-dev
wget -N https://github.com/mccdaq/uldaq/releases/download/v1.2.1/libuldaq-1.2.1.tar.bz2
tar -xvjf libuldaq-1.2.1.tar.bz2
cd libuldaq-1.2.1
./configure && make
sudo make install
```

## Serial communication

grant permissions to user to access serial port
```
sudo usermod -aG dialout $USER
# sudo reboot
```
## Nexline

install PI libraries
```
sudo apt install lbzip2
```
then in `PI-Software-Suite-C-990.CD1/Linux/PI_Application_Software-1.23.0.1-INSTALL/PI_Application_Software/`
```
sudo ./INSTALL
```

And in `PIPython-2.10.2.1-INSTALL/PIPython/PIPython-2.10.2.1/`
```
conda activate servo
python setup.py install
```

# run servo

Run the command
```
servo run
```

## copy defaults states

copy `state.json` in `~/local/states/`



