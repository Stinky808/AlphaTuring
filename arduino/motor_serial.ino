const int IN1 = 2;
const int IN2 = 3;
const int IN3 = 4;
const int IN4 = 5;

const int TRIG_PIN = 11;
const int ECHO_PIN = 12;

const int TOO_CLOSE_CM = 35;

String command = "";

unsigned long lastCommandTime = 0;
const unsigned long COMMAND_TIMEOUT = 700;

void setup() {
  Serial.begin(9600);

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  stopMotors();
}

void loop() {
  if (Serial.available()) {
    command = Serial.readStringUntil('\n');
    command.trim();

    lastCommandTime = millis();

    handleCommand(command);
  }

  if (millis() - lastCommandTime > COMMAND_TIMEOUT) {
    stopMotors();
  }
}

void handleCommand(String cmd) {
  int distance = readDistanceCm();

  if (cmd == "FORWARD") {
    if (distance < 0 || distance <= TOO_CLOSE_CM) {
      stopMotors();
      Serial.println("BLOCKED_TOO_CLOSE");
    } else {
      forward();
    }
  } 
  else if (cmd == "BACKWARD") {
    backward();
  } 
  else if (cmd == "LEFT") {
    left();
  } 
  else if (cmd == "RIGHT") {
    right();
  } 
  else if (cmd == "STOP") {
    stopMotors();
  } 
  else {
    stopMotors();
  }
}

int readDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);

  if (duration == 0) {
    return -1;
  }

  int distance = duration * 0.034 / 2;
  return distance;
}

void forward() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void backward() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void left() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void right() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void stopMotors() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}
