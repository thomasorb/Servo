# Servo
Servo for SpIOMM and THÉSÉE

## Install software

### Install python environment

#### Install Miniconda

https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2

#### Install servo env

download the [installation file](./servo.yml) then run:

```
conda env create -f servo.yml
```

### Install software
```
git clone https://github.com/thomasorb/Servo.git
conda activate servo
cd Servo
python servo.py install # for simple user but not for developper
```

#### developper only

Add module
```
# add to file: ~/miniconda3/envs/servo/lib/python3.12/site-packages/conda.pth
/path/to/Servo
```

Add command
```
ln -s /absolute/path/to/Servo/bin/servo ~/miniconda3/envs/servo/
```

## Install drivers

### NIT IR Camera drivers

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

### MCC DAQ drivers

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

### Serial communication

grant permissions to user to access serial port
```
sudo usermod -aG dialout $USER
# sudo reboot
```
### Nexline

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

## copy defaults states

copy `state.json` in `~/local/states/`


## run servo

Compile the set of C functions
```
python setup.py build_ext --inplace
```

Give rights to set negative niceness
```
sudo setcap cap_sys_nice+ep Servo/bin/servo
```

Run it :)
```
servo run
```




