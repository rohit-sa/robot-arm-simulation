#include "SerialComm.h"

SerialComm::SerialComm() {
  for (int i = 0; i < this->angleBufferSize; i++) {
    this->angleBuffer[i] = 0;
  }
}

SerialComm::~SerialComm() {
}

bool SerialComm::initSerialComm(uint baudRate = 115200) {
  Serial.begin(baudRate);
  return true;
}

bool SerialComm::getAngleData() {
  int bufferSize = this->angleBufferSize;
  memset(angleBuffer, 2, bufferSize);
  if (Serial.available() == bufferSize) {
    Serial.readBytes(angleBuffer, bufferSize);
    delay(50);
    return true;
  }
  return false;

}

