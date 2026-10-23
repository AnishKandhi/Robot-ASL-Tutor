This project features an American Sign Language fingerspelling robotic hand designed to help deaf learners learn ASL through a quiz format. The first part of the quiz prompts the user
to sign ASL letters into a web cam to translate a string of randomly generated English letters, and a AI vision model (Convolutional Neural Network) converts their signs into English letters
to measure accuracy. The correct answers are finally shown via the robot hand signing. The second part of the quiz starts with the robotic hand displaying ASL signs to the learner,
and the learner is tasked with entering the English-translated string of letters into the computer, and accuracy is scored appropriately. To train the CNN, we extracted landmarks from the user's hand
(x, y, z coordinates of 21 different locations) and passed them into the CNN for training, collecting 200+ frames manually per letter. The training and inference script was programmed using Python libraries 
OpenCV for web cam footage, MediaPipe for landmark extraction, and TensorFlow for building the CNN vision model. In order to spell using the robotic hand, a Rasberry Pi Pico was
programmed in microPython to command servos that used voltage to exert tension and cause torque on the fingers, relative to their joints. The computer and Rasberry Pi Pico interacted via serial ports connected by a 
usb serial cable. 
