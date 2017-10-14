/*
  Pythton GUI controlled robot arm
*/

#include <Servo.h>
#include "SerialComm.h"

SerialComm usb = SerialComm();
/*
  Servo pin initializations and angle limiters
  servo count starts from bottom to top
*/
int servo_1_angle = 90, servo_2_angle = 65, servo_3_angle = 90,
    servo_4_angle = 0, servo_5_angle = 0, gesture = 0;

Servo servo_1, servo_2, servo_3, servo_4, servo_5, servo_6;
const int servo_1_pin = 3, servo_2_pin = 4, servo_3_pin = 5, servo_4_pin = 6,
          servo_5_pin = 7, servo_6_pin = 8;
const int button_reset_pin = 12, button_fb_pin = 11;
int servo_1_min_angle = 0, servo_1_max_angle = 180;
int servo_2_min_angle = 0, servo_2_max_angle = 90;
int servo_3_min_angle = 0, servo_3_max_angle = 180;
int servo_4_min_angle = 0, servo_4_max_angle = 180;
int servo_5_min_angle = 0, servo_5_max_angle = 180;
int servo_6_min_angle = 50, servo_6_max_angle = 95;

int i = 0;
/*
  Setup for servos pins and pinmode
*/
void setup() {

  pinMode(servo_1_pin, OUTPUT);
  pinMode(servo_2_pin, OUTPUT);
  pinMode(servo_3_pin, OUTPUT);
  pinMode(servo_4_pin, OUTPUT);
  pinMode(servo_5_pin, OUTPUT);
  pinMode(servo_6_pin, OUTPUT);
  pinMode(button_reset_pin, INPUT);
  pinMode(button_fb_pin, INPUT);

  servo_1.attach(servo_1_pin);
  servo_2.attach(servo_2_pin);
  servo_3.attach(servo_3_pin);
  servo_4.attach(servo_4_pin);
  servo_5.attach(servo_5_pin);
  servo_6.attach(servo_6_pin);
  // Center the motors
  servo_1.write(90);
  servo_2.write(90);
  servo_3.write(90);
  servo_4.write(0);
  servo_5.write(90);
  servo_6.write(95);
  delay(15);

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  usb.initSerialComm(9600);
}

void loop() {
  /*
     Read servo angles and gesture from usb
     Byte values for all angles
  */
  if (usb.getAngleData()) {
    servo_1_angle = 0xFF & usb.angleBuffer[0];
    servo_2_angle = 0xFF & usb.angleBuffer[1];
    servo_3_angle = 0xFF & usb.angleBuffer[2];
    servo_4_angle = 0xFF & usb.angleBuffer[3];
    gesture =  0x0F & usb.angleBuffer[4];

  }
  /*
     Reset button works only to center the arm
     using all servos
  */
  if (digitalRead(button_reset_pin) == HIGH) {
    servo_1_angle = 90;
    servo_2_angle = 65;
    servo_3_angle = 90;
    servo_4_angle = 0;
    gesture = 0;
    digitalWrite(LED_BUILTIN, HIGH);
  }
  else {
    digitalWrite(LED_BUILTIN, LOW);
  }
  /*
     Allow grabber servo close action if feedback is inactive
     and gesture is 1
     else open arm if gesture is 0
  */
  if (gesture == 1 && digitalRead(button_fb_pin) == LOW) {
    servo_6.write(servo_6_max_angle);
  }
  else if (gesture == 0) {
    servo_6.write(servo_6_min_angle);
  }
  /*
    Update angles to servos
  */
  servo_1.write(servo_1_angle);
  servo_2.write(servo_2_angle);
  servo_5.write(90);
  servo_3.write(servo_3_angle);
  servo_4.write(servo_4_angle);
  delay(25);
}


