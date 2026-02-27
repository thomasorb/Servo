import logging
import time
import numpy as np
import traceback

from uldaq import (get_daq_device_inventory, DaqDevice, InterfaceType,
                   Range, AOutFlag)

from . import core
from . import config

log = logging.getLogger(__name__)


class DAQ(core.Worker):

    def __init__(self, data, events):
        super().__init__(data, events)
        
        self.daq_device = None
        
        interface_type = InterfaceType.ANY

        # Get descriptors for all of the available DAQ devices.
        devices = get_daq_device_inventory(interface_type)
        number_of_devices = len(devices)
        if number_of_devices == 0:
            raise RuntimeError('Error: No DAQ devices found')

        log.info(f'Found {number_of_devices} DAQ device(s):')
        for i in range(number_of_devices):
            log.info(f' [{i}] {devices[i].product_name} ({devices[i].unique_id})')

        self.daq_device = DaqDevice(devices[0])
        # Get AoDevice and AoInfo objects for the analog input subsystem
        self.ao_device = self.daq_device.get_ao_device()

        descriptor = self.daq_device.get_descriptor()
        log.info(f'Connecting to {descriptor.dev_string}')
        self.daq_device.connect(connection_code=0)

        ao_info = self.ao_device.get_info()
        log.info(f'accepted range {ao_info.get_ranges()}')
        log.info(f'{descriptor.dev_string} ready')
        log.info('MCC DAQ initialized')

        self.last_levels = None
        
    def loop_once(self):
        levels = self.data['DAQ.piezos_level'][:3]
        
        # smooth level change on piezos
        if self.last_levels is None:
            self.last_levels = levels

        else:
            new_levels = list()
            for ilevel, ilast_level in zip(levels, self.last_levels):
                if np.abs(ilevel - ilast_level) > config.DAQ_MAX_LEVEL_CHANGE:
                    if ilevel > ilast_level:
                        ilevel = ilast_level + config.DAQ_MAX_LEVEL_CHANGE
                    else:
                        ilevel = ilast_level - config.DAQ_MAX_LEVEL_CHANGE
                new_levels.append(ilevel)

            levels = new_levels

            self.last_levels = levels

            for (ichannel, ilevel) in zip(config.DAQ_PIEZO_CHANNELS, levels):
                
                self.ao_device.a_out(ichannel, Range.UNI10VOLTS,
                                     AOutFlag.DEFAULT, float(ilevel))
            

            self.data['DAQ.piezos_level_actual'][:3] = np.array(
                levels, dtype=config.DAQ_PIEZO_LEVELS_DTYPE)
            
        time.sleep(config.DAQ_LOOP_TIME)


    def cleanup(self):
        try:
            if self.daq_device is not None:
                # Disconnect from the DAQ device.
                if self.daq_device.is_connected() is not None:
                    self.daq_device.disconnect()
                    
                # Release the DAQ device resource.
                self.daq_device.release()

            log.info('cleanup ok')

        except Exception as e:
            log.error(f"Error at cleanup:\n {traceback.format_exc()}")

