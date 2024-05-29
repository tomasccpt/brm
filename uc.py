import serial
import time

PORT = 'COM3'

ROBOT = False
if ROBOT:
    ser = serial.Serial(PORT, 9600)
    last_timestamp = 0

    def send(values):
        global ser
        while True:
            try:
                ser.write(bytes(values))
                break
            except:
                time.sleep(0.1)
                print("Robot has disconnected. Re connecting...: ", values)
                ser.close()
                while True:
                    try:
                        ser = serial.Serial(PORT, 9600)
                        break
                    except:
                        time.sleep(0.1)
                        print("Didn't reconnect")
else:
    def send(values):
        time.sleep(0.1)
