import serial
import time

ROBOT = False
if ROBOT:
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
        global ser
        while True:
            try:
                ser.write(bytes(values))
                break
            except:
                time.sleep(0.1)
                print("morreu: ", values)
                ser.close()
                while True:
                    try:
                        ser = serial.Serial('COM3', 9600)
                        break
                    except:
                        time.sleep(0.1)
                        print("não renasceu")
else:
    def send(values):
        time.sleep(0.1)
