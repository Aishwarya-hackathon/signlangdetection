# Sign language Detection

Mode of communication for people with disabilities like hearing and speaking has always been controversial. Sign language has proven to be an incredibly beneficial method. It deals with recognizing gestures which are then comprehended as words or alphabets. The majority of the crowd are not often exposed to the world of sign language significantly. This causes issues for people with disabilities which results in deprived communication with the majority. The purpose of this work is to provide a real-time system which can convert our own set of signs into text. 

First, we create a database as an act of extending a step forward in this field and then various image preprocessing and feature extraction techniques were performed to obtain reasonable result. Initially as an easy approach we went ahead experimenting with just up to 5 different classes/words whose self-made images were fed into our CNN model resulting in a classification accuracy of 97%. As an addition to the work that we accomplished on static images, we created a live demo version of the project which can be run at a little less than 2 seconds per frame to classify signed hand gestures from any person.

[Download the video demo](final_video_1.mp4)

## Collecting dataset
![image](https://user-images.githubusercontent.com/74018041/121787111-e3829c80-cbe1-11eb-9353-f6c4ca2df4b6.png)

## Testing the Model
![image](https://user-images.githubusercontent.com/74018041/121787128-090fa600-cbe2-11eb-8a24-976dc1c79d94.png)



Result for HELLO 
![image](https://user-images.githubusercontent.com/74018041/121787208-6efc2d80-cbe2-11eb-9e0c-16067929a88f.png) 
Result for BYE
![image](https://user-images.githubusercontent.com/74018041/121787219-7cb1b300-cbe2-11eb-8d61-0528aafeef17.png)

Result for NO 
![image](https://user-images.githubusercontent.com/74018041/121787263-b71b5000-cbe2-11eb-93a6-33205e41efeb.png)
Result for OKAY
![image](https://user-images.githubusercontent.com/74018041/121787270-c0a4b800-cbe2-11eb-99b6-4750c0c56c10.png)
  

### Time Taken to detect = 0.922s & Accuracy of our model = 97%
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Research Paper Publication 
Published a research paper on real-time sign language recognition, developed a live demo for gesture classification, and presented the work at an IEEE International Conference on Distributed Computing, VLSI, Electrical Circuits and Robotics (DISCOVER).
https://ieeexplore.ieee.org/document/9663629/references#references
