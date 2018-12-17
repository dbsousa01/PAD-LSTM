# PAD-LSTM
This work was done to achieve the Master Degree in the course of Electrical and Computer Engineering in Instituto Superior Técnico [(IST)](https://tecnico.ulisboa.pt/pt/). Thesis can be read [here](https://www.dropbox.com/s/0nd4ce7m6br5ds2/Thesis.pdf?dl=0)
(article is still in development)

## Presentation Attack Detection using a Long Short Term Memory Layer

The work was done in Python, using the Keras library to develop a Deep Learning algorithm (CNN + LSTM) in order to indentify attacks to face recognition systems

### Abstract:
Face recognition systems are increasingly important in today’s society, being mainly employed as a security measure. Everyday items, such as mobile phones and laptops, or more crucial security systems, such as airport access control, are good examples of its usage. Due to its popularity, these biometric systems are vulnerable to a wide range of attacks, which are becoming more and more complex. Therefore the development of effective counter-measures is necessary. The objective of this thesis is to develop a tool which detects intrusions at the sensor level, known as Presentation Attacks (PA). For this, state of the art contributions are reviewed in order to understand their main disadvantages and possibilities of improvement. An approach based on transfer learning using a pre-trained Convolutional Neural Network (CNN) model is presented. This network is then adapted to the problem and several steps are taken to optimise it accordingly. A novel approach is proposed by implementing a layer that performs video analysis for action classification, known as Long Short Term Memory (LSTM). The proposed solution achieves a half total error rate (HTER) of 1.09% in the Replay-Attack database. Finally, a conclusion is made about the detection of attacks to facial recognition systems and why is it still an open problem, even though state of the art methods show a high performance in such demanding databases.


cnn_lstm.py - The main kernel where the CNN + LSTM training happens  
fusion.py - Architecture fusion which lead to worse results  
VideoCapture.py - Used to capture frames from the database videos  
test_lstm.py - Testing of the neural network

The VGG-Face model was built according to this blogpost: https://aboveintelligent.com/face-recognition-with-keras-and-opencv-2baf2a83b799 The weights of the trained model can also be found there.
