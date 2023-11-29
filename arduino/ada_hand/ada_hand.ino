#include <ros.h>
#include <std_msgs/Int16MultiArray.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
#include <FingerLib.h>

#define ALMOND_BOARD
#define NUM_FINGERS 5



int handFlag = RIGHT;
Finger finger[NUM_FINGERS];

// start ros node
ros::NodeHandle  nh;


// toggle all motors and maintain the current position
void EnableMotors(bool en){
  for (int i = 0; i < NUM_FINGERS; i++){
    finger[i].motorEnable(en);
  }
}


// receive a 5 dimension array to change finger positions
void setFingerPos(float* fingerPosArray){
  // maximum of each finger for safety reasons
  float fingerMaxPosition[5] = {970, 940, 950, 920, 950};
  
  for(int i = 0; i < NUM_FINGERS; i++){
    int fingerPosInt = fingerPosArray[i] * fingerMaxPosition[i];
    finger[i].writePos(fingerPosInt);
  };
}


// choose grasp type of the hand
void setGraspType(String msg){
  if(msg == "Pinch"){
    float fingerPosArray[5] = {1, 1, 1, 0, 0};
    setFingerPos(fingerPosArray);
  }
  else if(msg == "Power"){
    float fingerPosArray[5] = {1, 1, 1, 1, 1};
    setFingerPos(fingerPosArray);
  }
  else if(msg == "Open"){
    float fingerPosArray[5] = {0, 0, 0, 0, 0};
    setFingerPos(fingerPosArray);
  }
}


std_msgs::Int16MultiArray pose;

// update feedback topic data
void updateFeedback(){
  int16_t fingerPosArray[NUM_FINGERS];
  for(int i = 0; i < NUM_FINGERS; i++){
    fingerPosArray[i] = finger[i].readPos();
  };
  pose.data = fingerPosArray;
}


// feedback topic
ros::Publisher feedback_pub("feedback", &pose);

// grasp type topic
void cbGrasp( const std_msgs::String& msg){
  setGraspType(msg.data);
}
ros::Subscriber<std_msgs::String> grasp_sub("grasp", cbGrasp);

// enable topic
void cbEnableMotors(const std_msgs::Bool& msg){
  EnableMotors(msg.data);
}
ros::Subscriber<std_msgs::Bool> enable_motors_sub("enable_motors", cbEnableMotors);



void setup(){
  pinAssignment();
  
  nh.initNode();
  nh.advertise(feedback_pub);
  nh.subscribe(grasp_sub);
  nh.subscribe(enable_motors_sub);
  
  pose.data_length = NUM_FINGERS;
}

void loop(){
   // update publisher
   updateFeedback();
    
   feedback_pub.publish( &pose );
    
   nh.spinOnce();
   delay(1);

}
