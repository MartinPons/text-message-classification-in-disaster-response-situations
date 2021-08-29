# Text message classification in disaster response situations


## DPROBLEM DEFINITION

When some type of disaster occurs, like an earthquake, flooding, etc. Millions of communications are generated through various channels, including in social media, searching for some kind of help. In order for the appropiate organization to take care of these messages, they need to be classified in different categories. Disaster response **organizations have a limited capacity to filter the most important messages**. An automated model that labels each message can save resources and speed up the time response in those crucial moments after the disaster has occurred.


## DATA UNDERSTANDING

The data comes from **[Figure Eight](https://appen.com/)**. It consists in a collection of 26248 mmessages manually labeled in 36 categories which include things like weather, cold, eathquake, shelter... A message can be labeled in more than one category.

There are **two datasets**: one with the **labeled categories** (one column for each category), and the other with the **content of the messages**, the message in the original language (when available) and the genre which can be one of *direct*, *news* and *sociial*, specifying the channel the message was sent through.


## DATA PREPARATION

The messages are cleaned and a **bag of words** is created by tokenizing the messages. Stop words are previously removed.


## MODELING

A **classification model** has been developed using **python**, which labels the messages into one or more category out of 36. The model is a **multi-ouptut model**, that is, a message can be classified with more than one label. A Random Forest was trained for each one of these categories, using the aforementioned multi-output model. Different evaluation meassures are obtained for everyone of the 36 categories. The **F1 score ranges from 0.77 to 1**. There is a list of the varaibles used in the model below.

- Tfidf scoring
- Length of the message measured in number of characters
- A flag indicating wheter the message includes exclamation points.


## WEBB AP

A very basic web app was developed to demonstrate the practical utility of the model: the user introduces a text in the box and the tags predicted are returned. The web also includes some basic visualizations to give insight into some of the training data features.
 
## FILE STRUCTURE

- **requirements.txt** a file with all the packages installed in the environment where the model was executed.
- **app**: contains the html code for the app and a python scrpit, run.py, to launch the app.
- **data**: contains the following files.
	- **distaster_messages.csv**: text messages sent through social media.
	- **disaster_categories.csv**: correspondent labelling of the messages into 36 possilbe categories.
	- **process_data.py**: python script with the data preprocessing. Creates a database to save the processed data with the name DisasterResponse.db into the same folder.

- **models**
	- **train_classifier.py**: script that trains and evaluates the model. Also saves the model into the file "classifier.plk" into the same folder.

## HOW TO RUN THE SCRIPTS

**To run the process_data.py script**, type python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db into the console. This generates the data base with the processed data.

**To run the train_classifier.py** script, type python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl into the console. This generates the classifier object that can be use in the web app.

**To run the app** go to the app directory and type python run.app.

