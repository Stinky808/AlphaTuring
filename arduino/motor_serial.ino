#include <Servo.h>

const int IN1 = 2;
const int IN2 = 3;
const int IN3 = 4;
const int IN4 = 5;
const int ENA = 9;
const int ENB = 10;
const int TRIG_PIN = 11;
const int ECHO_PIN = 12;
const int SERVO_PIN = 6;
const int TOO_CLOSE_CM = 35;

int motorSpeed = 180;

// --- Servo pan state ---
Servo panServo;
int servoAngle = 90;
int servoStep = 2;          // Degrees per update (higher = faster pan)
unsigned long lastServoTime = 0;
const int SERVO_INTERVAL = 20;  // ms between steps (lower = faster pan)
const int SERVO_MIN = 20;
const int SERVO_MAX = 160;

String command = "";
unsigned long lastCommandTime = 0;
const unsigned long COMMAND_TIMEOUT = 700;

void setup() {
  Serial.begin(9600);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  panServo.attach(SERVO_PIN);
  panServo.write(servoAngle);

  stopMotors();
}

void loop() {
  // Handle incoming serial commands
  if (Serial.available()) {
    command = Serial.readStringUntil('\n');
    command.trim();
    lastCommandTime = millis();
    handleCommand(command);
  }

  // Motor timeout
  if (millis() - lastCommandTime > COMMAND_TIMEOUT) {
    stopMotors();
  }

  // Non-blocking servo pan
  updateServo();
}

void updateServo() {
  if (millis() - lastServoTime >= SERVO_INTERVAL) {
    lastServoTime = millis();
    servoAngle += servoStep;

    // Bounce at limits
    if (servoAngle >= SERVO_MAX) {
      servoAngle = SERVO_MAX;
      servoStep = -servoStep;
    } else if (servoAngle <= SERVO_MIN) {
      servoAngle = SERVO_MIN;
      servoStep = -servoStep;
    }

    panServo.write(servoAngle);
  }
}

void handleCommand(String cmd) {
  if (cmd.startsWith("SPEED:")) {
    int val = cmd.substring(6).toInt();
    motorSpeed = constrain(val, 0, 255);
    Serial.print("SPEED_SET:");
    Serial.println(motorSpeed);
    return;
  }

  int distance = readDistanceCm();

  if (cmd == "FORWARD") {
    if (distance < 0 || distance <= TOO_CLOSE_CM) {
      stopMotors();
      Serial.println("BLOCKED_TOO_CLOSE");
    } else {
      forward();
    }
  } 
  else if (cmd == "BACKWARD") { backward(); }
  else if (cmd == "LEFT")     { left();     }
  else if (cmd == "RIGHT")    { right();    }
  else                        { stopMotors(); }
}

// --- Motor helpers ---

void setSpeed(int speed) {
  analogWrite(ENA, speed);
  analogWrite(ENB, speed);
}

void forward() {
  setSpeed(motorSpeed);
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}

void backward() {
  setSpeed(motorSpeed);
  digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);
}

void left() {
  setSpeed(motorSpeed);
  digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}

void right() {
  setSpeed(motorSpeed);
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);
}

void stopMotors() {
  setSpeed(0);
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}

// --- Ultrasonic ---

int readDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  if (duration == 0) return -1;
  return duration * 0.034 / 2;
}
