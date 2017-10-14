#ifndef SerialComm_H
#define SerialComm_H

#include "Arduino.h"

typedef unsigned int uint;

class SerialComm{
  public:
    SerialComm();
    ~SerialComm();
    bool initSerialComm(uint baudRate = 115200);
    bool getAngleData();
    unsigned char angleBuffer[5];
  private:
    int angleBufferSize = 5;
};

#endif
