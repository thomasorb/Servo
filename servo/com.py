# -*- coding: utf-8 -*-
"""
serial_comm.py — Binary serial communication worker for Servo.

Key features
------------
- Parses binary TLV commands with XOR checksum
- Triggers Servo events (start/stop/open_loop/close_loop/move_to_opd)
- Publishes compact STATUS frames at high rate (>= 1 kHz)
- Sends ACK/NACK for every received command
- Non-blocking RX thread and preallocated TX buffer to minimize jitter

Dependencies
------------
- pyserial

Binary protocol
---------------
RX (Host -> Servo):
    [STX_RX=0xAA] [TYPE] [LEN] [PAYLOAD] [CHK]
    CHK = XOR of TYPE..PAYLOAD (STX excluded). LEN = payload size.

    Types:
        0x01 : START        (LEN=0)
        0x02 : STOP         (LEN=0)
        0x03 : OPEN_LOOP    (LEN=0)
        0x04 : CLOSE_LOOP   (LEN=0)
        0x05 : MOVE_TO_OPD  (LEN=4, payload <float32 LE>)

TX (Servo -> Host):
    [STX_TX=0x55] [TYPE] [LEN] [PAYLOAD] [CHK]

    Types:
        0x10 : STATUS (LEN=5, payload = <float32 opd_mean><uint8 servo_state>)
        0x11 : ACK    (LEN=1, payload = <uint8 cmd_type_echo>)
        0x12 : NACK   (LEN=2, payload = <uint8 cmd_type_echo><uint8 err_code>)
            err_code: 0x01=BAD_CHECKSUM, 0x02=MALFORMED, 0x03=UNKNOWN

Recommended link settings
-------------------------
- Baudrate: 1 000 000 (or 921600 / 2 000 000 depending on hardware)
- STATUS at 1 kHz: 9 bytes/frame (header+payload+chk) -> ~9 kB/s

Integration
-----------
from . import serial_comm
self.start_worker(serial_comm.SerialComm, 0,
                  port="/dev/ttyUSB0", baudrate=1_000_000, status_rate_hz=1000)
"""

import time
import threading
import logging
import struct
import serial  # pyserial

from . import core

log = logging.getLogger(__name__)

# Protocol constants
STX_RX = 0xAA  # Host -> Servo
STX_TX = 0x55  # Servo -> Host

# RX command types
CMD_START       = 0x01
CMD_STOP        = 0x02
CMD_OPEN_LOOP   = 0x03
CMD_CLOSE_LOOP  = 0x04
CMD_MOVE_TO_OPD = 0x05
CMD_NORMALIZE   = 0x06

# TX message types
TX_STATUS = 0x10
TX_ACK    = 0x11
TX_NACK   = 0x12

# NACK error codes
ERR_BAD_CHECKSUM = 0x01
ERR_MALFORMED    = 0x02
ERR_UNKNOWN      = 0x03


def xor_checksum(data: bytes) -> int:
    """Fast XOR checksum over the given bytes."""
    c = 0
    for b in data:
        c ^= b
    return c & 0xFF


class SerialComm(core.Worker):
    """
    Serial communication worker: receives binary commands and publishes a fast STATUS stream.

    __init__ parameters
    -------------------
    port : str           e.g., "/dev/ttyUSB0" or "COM5"
    baudrate : int       e.g., 1_000_000
    status_rate_hz : int e.g., 1000 (1 kHz)
    """

    def __init__(self, data, events,
                 port="/dev/ttyUSB0",
                 baudrate=1_000_000,
                 status_rate_hz=1000):
        super().__init__(data, events)
        self.data = data
        self.events = events

        self.port = port
        self.baudrate = int(baudrate)
        self.period = 1.0 / float(status_rate_hz)

        # Open non-blocking serial port for low latency
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.0,          # non-blocking reads
                write_timeout=0.0,    # non-blocking writes
                inter_byte_timeout=None
            )
        except Exception as e:
            log.error(f"Failed to open serial port {self.port}: {e}. If the permission was undenied don't forget to add dialout group to your user and re-login with'sudo usermod -aG dialout $USER.")
            raise

        # RX state
        self._rx_buf = bytearray()
        self._keep_running = True

        # Dedicated RX thread (so TX loop can keep precise timing)
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

        # Pre-allocated STATUS frame: [STX_TX][TYPE][LEN][payload(5)][CHK]
        # payload: <float32 opd><uint8 state>  -> LEN=5
        self._status_hdr = bytes([STX_TX, TX_STATUS, 5])
        self._tx_frame = bytearray(3 + 5 + 1)  # header + payload + chk
        self._tx_frame[0:3] = self._status_hdr

        log.info(f"SerialComm on {self.port} @ {self.baudrate} baud, "
                 f"status {int(1.0/self.period)} Hz")

    # -------------------------------------------------------------------------
    # RX path: read and parse TLV with STX resync
    # -------------------------------------------------------------------------
    def _rx_loop(self):
        while self._keep_running:
            try:
                # Read small chunks to keep latency low
                chunk = self.ser.read(128)
                if chunk:
                    self._rx_buf.extend(chunk)
                    self._parse_rx_buffer()
                else:
                    # Relax CPU when no data (short sleep keeps latency small)
                    time.sleep(0.0005)
            except Exception as e:
                log.error(f"Serial RX error: {e}")
                time.sleep(0.005)

    def _parse_rx_buffer(self):
        buf = self._rx_buf
        while True:
            # Minimum frame size: STX + TYPE + LEN + CHK = 4
            if len(buf) < 4:
                return

            # Find STX (resync if necessary)
            if buf[0] != STX_RX:
                del buf[0]
                continue

            if len(buf) < 3:
                return

            cmd_type = buf[1]
            length = buf[2]
            total_len = 3 + length + 1
            if len(buf) < total_len:
                return

            payload = buf[3:3+length]
            chk = buf[3+length]

            # Verify XOR checksum over TYPE..PAYLOAD (exclude STX)
            if xor_checksum(buf[1:3+length]) != chk:
                self._send_nack(cmd_type, ERR_BAD_CHECKSUM)
                del buf[:total_len]
                log.error('bad checksum')
                continue

            # Process valid command
            self._process_command(cmd_type, payload)
            self._send_ack(cmd_type)

            # Consume frame
            del buf[:total_len]

    # -------------------------------------------------------------------------
    # Command handling
    # -------------------------------------------------------------------------
    def _process_command(self, cmd_type: int, payload: bytes):
        try:
            if cmd_type == CMD_START:
                if len(payload) != 0:
                    self._send_nack(cmd_type, ERR_MALFORMED); return
                self.events["Servo.start"].set()
                return

            if cmd_type == CMD_STOP:
                if len(payload) != 0:
                    self._send_nack(cmd_type, ERR_MALFORMED); return
                self.events["Servo.stop"].set()
                return

            if cmd_type == CMD_OPEN_LOOP:
                if len(payload) != 0:
                    self._send_nack(cmd_type, ERR_MALFORMED); return
                self.events["Servo.open_loop"].set()
                return

            if cmd_type == CMD_CLOSE_LOOP:
                if len(payload) != 0:
                    self._send_nack(cmd_type, ERR_MALFORMED); return
                self.events["Servo.close_loop"].set()
                return
            
            if cmd_type == CMD_NORMALIZE:
                if len(payload) != 0:
                    self._send_nack(cmd_type, ERR_MALFORMED); return
                self.events["Servo.normalize"].set()
                return

            if cmd_type == CMD_MOVE_TO_OPD:
                if len(payload) != 4:
                    self._send_nack(cmd_type, ERR_MALFORMED); return
                # float32 little-endian
                opd_target = struct.unpack("<f", payload)[0]
                # Write target then trigger event
                self.data["Servo.opd_target"][0] = float(opd_target)
                self.events["Servo.move_to_opd"].set()
                return

            # Unknown command
            self._send_nack(cmd_type, ERR_UNKNOWN)

        except Exception as e:
            log.error(f"Error processing command 0x{cmd_type:02X}: {e}")
            self._send_nack(cmd_type, ERR_MALFORMED)

    # -------------------------------------------------------------------------
    # TX path: STATUS (high rate) + ACK/NACK (sporadic)
    # -------------------------------------------------------------------------
    def _send_status(self):
        """
        STATUS frame layout:
            [0]=STX_TX(0x55)
            [1]=TYPE(0x10)
            [2]=LEN(5)
            [3..6]=<float32 opd_mean>
            [7]=<uint8 servo_state>
            [8]=CHK (XOR over bytes [1..7])
        """
        # Read shared data (robust defaults if not ready yet)
        try:
            opd = float(self.data["IRCamera.mean_opd"][0])
        except Exception:
            opd = float("nan")
        try:
            state = int(self.data["Servo.state"][0])
        except Exception:
            state = 0

        # Pack payload into preallocated frame
        struct.pack_into("<f", self._tx_frame, 3, opd)  # bytes 3..6
        self._tx_frame[7] = state & 0xFF

        # Compute checksum over TYPE+LEN+PAYLOAD (bytes 1..7)
        c = xor_checksum(self._tx_frame[1:8])
        self._tx_frame[8] = c

        # Non-blocking write
        try:
            self.ser.write(self._tx_frame)
        except Exception:
            # Avoid logging at high frequency to keep timing stable
            pass

    def _send_ack(self, cmd_type: int):
        """
        ACK frame:
            [STX_TX][TX_ACK][LEN=1][<u8 cmd_type>][CHK]
        """
        payload = bytes([cmd_type & 0xFF])
        header = bytes([STX_TX, TX_ACK, 1])
        chk = xor_checksum(bytes([TX_ACK, 1]) + payload)
        frame = header + payload + bytes([chk])
        try:
            self.ser.write(frame)
        except Exception:
            pass

    def _send_nack(self, cmd_type: int, err_code: int):
        """
        NACK frame:
            [STX_TX][TX_NACK][LEN=2][<u8 cmd_type><u8 err_code>][CHK]
        """
        payload = bytes([(cmd_type & 0xFF), (err_code & 0xFF)])
        header = bytes([STX_TX, TX_NACK, 2])
        chk = xor_checksum(bytes([TX_NACK, 2]) + payload)
        frame = header + payload + bytes([chk])
        try:
            self.ser.write(frame)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Worker API
    # -------------------------------------------------------------------------
    def loop_once(self):
        """
        Main loop: send STATUS at the target rate.
        RX runs on a dedicated thread.
        """
        t0 = time.perf_counter()
        self._send_status()
        dt = time.perf_counter() - t0
        delay = self.period - dt
        if delay > 0:
            time.sleep(delay)

    def stop(self):
        """Graceful stop of the serial worker."""
        log.info("Stopping SerialComm")
        self._keep_running = False
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
