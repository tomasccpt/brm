import serial
import time

ser = serial.Serial('COM3', 9600)
last_timestamp = 0

def send(values):
    """
    global last_timestamp
    # Works as long as each list value is a single byte (-127 <= 0 <= 255).
    while True:
        t = time.perf_counter_ns()
        if t - last_timestamp > 0.8e6:
            break
        time.sleep(0)
    ser.write(bytes(values))
    #ser.flush()
    last_timestamp = time.perf_counter_ns()
    """
    while True:
        try:
            ser.write(bytes(values))
            break
        except:
            time.sleep(0.1)        
    
    
