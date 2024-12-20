import os
import librosa
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from python_speech_features import mfcc
import random
import operator

app = Flask(__name__)

# Define the correct directory where your audio files are located
directory = r"C:\Users\hp\Downloads\Telegram Desktop\genres_original\genres_original"

# Load the dataset
dataset = []


def loadDataset(filename, split, trset, teset):
    with open(filename, 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                break
    for item in dataset:
        if random.random() < split:
            trset.append(item)
        else:
            teset.append(item)


# Distance calculation function
def distance(instance1, instance2, k):
    mm1, cm1 = instance1[0], instance1[1]
    mm2, cm2 = instance2[0], instance2[1]
    dist = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    dist += (np.dot(np.dot((mm2 - mm1).T, np.linalg.inv(cm2)), mm2 - mm1))
    dist += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    dist -= k
    return dist


# Function to get neighbors
def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = [distances[x][0] for x in range(k)]
    return neighbors


# Function to classify the genre
def nearestclass(neighbors):
    classVote = {}
    for response in neighbors:
        classVote[response] = classVote.get(response, 0) + 1
    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]


# Load training and test set
trainingSet = []
testSet = []
loadDataset('mydataset.dat', 0.68, trainingSet, testSet)

# Load genre mapping
results = {}
i = 1
for folder in os.listdir(directory):
    results[i] = folder
    i += 1


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Save the uploaded file to the static/uploads folder
        upload_path = os.path.join('static', 'uploads', file.filename)
        if not os.path.exists(os.path.dirname(upload_path)):
            os.makedirs(os.path.dirname(upload_path))

        file.save(upload_path)

        # Process the file to extract features
        sig, rate = librosa.load(upload_path, sr=None)
        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix, covariance)

        # Classify the genre
        neighbors = getNeighbors(trainingSet, feature, 5)
        predicted_genre = nearestclass(neighbors)
        genre = results[predicted_genre]

        return jsonify({'predicted_genre': genre})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
