# AMPLIFY – Mood-Based Song Recommendation System

# Project Overview

AMPLIFY is a Mood-Based Song Recommendation System

* This project was completed as part of my final year undergraduate project in May 2022 along with my teammate Harshita Sharma.

This project essentially uses facial emotion detection to get the user's mood, which is captured through the webcam. The system then recommends the user a music playlist based on user's detected mood. 

Performs real-time mood detection using Machine Learning Techniques (CNN) and recommends a playlist according to the detected mood. The Model has roughly 80% accuracy. Uses 5 moods namely: happy, sad, neutral, surprised, angry


## Features

- User registration and login
- Facial emotion classification into five categories: Angry, Happy, Neutral, Sad, and Surprise.
- Display user's emotion
- Mood based songs and playlists recommended to user
- Music player controls to play and select songs.

Dataset: 

We used the Kaggle dataset to train the model, which included many facial photos expressing diverse emotions. We also utilized this trained model in conjunction with the Python OpenCV package to analyze the images and match them to the most accurate mood using the machine learning model.

The Dataset that we have chosen has been trained using convolution neural networks.

Source of reference data set : https://www.kaggle.com/msambare/fer2013

The task is to categorize each face based on the emotion shown in the facial expression into one of five categories (0=Angry, 1=Happy, 2=Neutral, 3=Sad, 4=Surprise). The training set consists of 28,709 examples and the public test set consists of 3,589 examples.This trained dataset is stored in the format of a h5 file. H5 is a Hierarchical Data Format(HDF) that is used to store enormous amounts of data. It is used to hold multidimensionalarrays containing vast amounts of data. The format is primarily intended to hold well-organized scientific data for easy retrieval and analysis. Hence the trained set is stored withthe name Emotion_little_vgg.h5 .

Machine Learning Model (CNN):

The machine learning model uses concepts of convolutional networks to detect the faces in the camera stream. After detection it analyses the frames in the stream and based on the training data set predicts the mood of that picture The input to the model that is an image was given using the REST API made in Django
and the list of songs along with the detected mood was sent back as a response. For handling so many states (songs list, video feed, many components on the home page) we decided to use React.js. Face detection can be thought of as such a problem where we detect human faces in an
image. There may be slight differences in the faces of humans but overall, it is safe to say that there are certain features that are associated with all the human faces

There are 3 steps for face detection:

1. Face detection

2. Feature extraction using face embedding

3. Facial recognition


System Architecture

When a user first visits the website, the main screen appears, which includes the AMPLIFY button for taking a picture. When the user presses the "AMPLIFY" button, the camera, which was already open, takes a photo. 

This image is fed into a facial recognition algorithm. If no face is discovered or 20 numerous faces are recognized,the user is given an appropriate error message. 

The photo is sent as input to the mood detection module after successful single face detection. 

The user is shown the detected mood, following which the "play tunes" button is enabled. 

On the playlist screen, as shown above, the appropriate playlist for the detected mood is provided, and the user can select and play the song of their choice. The complete playlist will be displayed as a drop-down menu, with their posters rotating in a queue.



Music Recommendation Module:

The image clicked is sent as an input to the REST API of Django , which then helps us fetch the songs list along with the detected mood to the screen. 


The songs have been manually added through an initial user account created in the Django administration with filters of song mood. The name of the song, the artist, the duration and the mood are all manually added to the database for each song. For this project, we have used the default sqlite that comes with Django as our RDBMS.


Integration and other features:

We choose React.js to handle so many states (music list, video feed, and several components on the home page).The entire backend has been built using Django and
Tensorflow and their respective modules. This project contains additional features,such as the ability to track people who are using the app. The users can see who are the other users registered and follow them. All the models have been declared and clearly defined in python files. A project was developed using React, Django, and Tensorflow, and mp3 music was uploaded to the storage portion. These songs are organized by the mood in Django's real-time database section, and Tensorflow assists us in detecting moods by accurately classifying emotions. React.js was used to construct an adequate UI for the website, and the program was then tested to find any flaws.


3.5. System Requirements

Hardware Requirements:

The following hardware is required for this project:
• A minimum of 4 Gigabytes (GB) of RAM (for processing) is required.
• Webcam (for laptop/desktop testing)
• Camera with a minimum resolution of 16 Megapixels (MP) (for testing on device)
• 30 MB of available memory (approximate value)

Software Requirements:

• Python 3.6
• Javascript
• React.js
• Node.js
• Keras
• Tensorflow
• Flask
*All other dependancies are listed in detail in the requirement.txt file.

Algorithm for Mood Detection:

1. Import the required libraries and dataset for building the network (pandas , numpy, keras and so on)
2. Divide the pixels into training and validation lists
3. Use the CNN model for mood detection and convert it into hdf5 file so that you do not have to train the network again.
4. In our model we have declared 5 moods: happy, angry, sad, neutral and surprised.
5. Bounding box will be drawn over the detected face , convert RGB images to greyscale and classify them in real time based on assigned moods.
6. Mood is displayed on screen accordingly, if no mood detected error message is displayed accordingly.


Algorithm for Music Recommendation:

1. Add the music manually to the Django database and tag each song with a mood
2. Get the detected mood as input
3. Depending on the identified mood , access the playlist accordingly from the database
4. Send the playlist to the frontend for access by the user.


Procedure for setting up the development environment:

>Install dependencies for nodes
>Python dependancies are installed (in a virtualenv)
>Using npm start and python manage.py runserver, start both servers.
>Create an admin user to upload the songs to and monitor the process. Add a mood to each song.
>Then, in your browser, navigate to your React dev URL and click Amplify under the
>Your facial expression is used to make music recommendations.

Tech stack:

- Frontend: React, React Router, Bootstrap 4, Axios, react-webcam, react-jinke-music-player
- Backend: Django, Django REST Framework (token auth)
- ML: Keras / TensorFlow, OpenCV

Project structure:

amplify/
├── frontend/
│   ├── public/index.html
│   └── src/            # App, MainApp, LoginForm, SignupForm, FriendsApp
└── backend/
    ├── manage.py
    ├── api/            # models, views, upload_songs management command
    └── amplifymodel/   # recog3.py (inference), train_model.py (CNN training)


Setup:

Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install django djangorestframework keras tensorflow opencv-python numpy pillow
python manage.py migrate
python manage.py runserver # http://localhost:8000
```

Frontend

```bash
cd frontend
npm install
npm start

The dataset used in Machine Learning is divided into three categories:

• Training dataset: the data used to fit the model.

• Validation dataset: the data used to validate the generalisation ability of the model or for early stopping, during the training process.

• Testing dataset: the data used to for other purposes other than training and validating.

Results:

We used 80% of the dataset for training, 10% for validation, and 10% for testing in our model. 
We next compared our training accuracy to our validation accuracy to determine our model's prediction certainty. 
We trained it for 25 epochs and achieved an accuracy of approximately 80%. 