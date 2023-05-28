#include <SoftwareSerial.h>
SoftwareSerial hc05(8,9);
String cmd="";
int enA = 2;
int in1 = 3;
int in2 = 4;
// motor two
int enB = 7;
int in3 = 5;
int in4 = 6;
int isforward = 1;
void setup()
{
  //Initialize Serial Monitor
  Serial.begin(9600);
  hc05.begin(9600);
  // set all the motor control pins to outputs
  pinMode(enA, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
 
  analogWrite(enA, 255);
  analogWrite(enB, 255); 
 
}
 
void loop()
{ 
  while(hc05.available()>0){
  cmd+=(char)hc05.read();
  }
  //control speed 
  //Select function with cmd
if(cmd!=""){
Serial.print("Command recieved : ");
Serial.println(cmd);
// We expect ON or OFF from bluetooth
if(cmd=="forward"){
  if (isforward==1){
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
    isforward = 0;
    
  }
  else {
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    digitalWrite(in3, LOW);
    digitalWrite(in4, LOW);
    isforward = 1;

  }
  }
if(cmd=="backwards"){ 
 digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
  isforward = 1;}
if(cmd=="right"){  
 digitalWrite(in1,HIGH ); 
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
  isforward = 1;}
if(cmd=="left"){
 digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
  isforward = 1;}
if(cmd=="none"){ 
 digitalWrite(in1, LOW); 
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
  isforward = 1;}
cmd=""; //reset cmd
}
delay(100);
}
