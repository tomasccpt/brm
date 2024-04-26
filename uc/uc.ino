#include <Servo.h>

#define len(x) (sizeof(x) / sizeof(*(x)))
#define clamp(min, x, max) ((x)<(min)?(min):(x)>(max)?(max):x)

Servo Servos[4];
int ReadVal[4] = { 0 };
int PastVal[4] = { 0 };
int Ports[4] = { 4, 5, 6, 7 };
int MinAngle[4] = { 10, 10, 10, 100 };
int MaxAngle[4] = { 170, 170, 170, 180 };

void setup() {
  Serial.begin(9600);

  for (size_t i = 0; i < len(Servos); ++i) {
    Servos[i].attach(Ports[i]);
  }
}

void loop() {
  char readBuf[sizeof(unsigned char)*len(ReadVal)];
  size_t read = Serial.readBytes(readBuf, sizeof(readBuf));
  bool unsynced = false;

  if (read) {
    unsynced = true;
    for (size_t i = 0; i < len(ReadVal); ++i) {
      ReadVal[i] = (int)*(unsigned char*)(readBuf + i);
    }
  }
  
  while (unsynced) {
    unsynced = false;
    for (size_t i = 0; i < len(Servos); ++i) {
      if ((ReadVal[i] - PastVal[i]) > 0) {
        PastVal[i] += 1;
        Servos[i].write(PastVal[i]);
        unsynced = true;
      } else if ((ReadVal[i] - PastVal[i]) < 0) {
        PastVal[i] -= 1;
        Servos[i].write(PastVal[i]);
        unsynced = true;
      }
    }
  }
}
