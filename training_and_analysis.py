import os
import pickle
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics as mt
from keras.models import model_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory

def get_training_data():
    sfs = {
    'models_path' : input("Type the directory where the models are stored: "),
    'output_path' : input("Type the directory where you want to save the models and the history of each model: "),
    'dataset' : input("Type the directory where the dataset is stored (it has to be separated into classes, and not splitted in train, test or valid subsets): "),
    'epochs' : int(input("Type the number of epochs you want to train the models: ")),
    'GPU' : input("Do you want to use GPU? (y/n or any key): ")}

    for label, value in sfs.items():
        if label in ['models_path', 'output_path', 'dataset']:
            check_dir(sfs[label])
            sfs[label] = check_dir(sfs[label])
        if label == 'GPU' and value == 'y':
            sfs[label] = True
        else:
            sfs[label] = False
    return sfs

def get_analysis_data():
    framework = input("Type the framework you used to train the models (tf or hf): ")
    sfs = {}

    if framework.lower() == 'tf':
        sfs = {
        "framework" : "tf",
        "models_path" : input("Type the directory where the models are stored, separated by commas: "),
        "folders" : input("Type the directories where the folders with the images are stored: "),
        "dataset_class" : input("Type 1 if the dataset is full of positive class images, or 0 if it's full of negative images: "),
        "image_height" : input("Type the height of the images: "),
        "image_width" : input("Type the width of the images: "),
        "output_path" : input("Type the directory where you want to save the results: ")}

    elif framework.lower() == 'hf':
        sfs = {
            "framework" : "hf",
            "transformers names" : input("Type the names of the transformers you want to use, separated by commas: "),
            "folders" : input("Type the directories where the folders with the images are stored, separated by commas: "),
            "dataset_class" : input("Type 1 if the dataset is full of positive class images, or 0 if it's full of negative images: "),
            "image_height" : input("Type the desired height of the images: "),
            "image_width" : input("Type the desired width of the images: "),
            "output_path" : input("Type the directory where you want to save the results: ")}
    return sfs


#it takes the path to the dataset, the batch size, the image size, the train size and 
#the valid size as inputs, and returns the train, test and valid subsets
def dataset(datadir, batch_size = 32, image_size = (224, 224), train_size = .6, valid_size = .2):
    data =  image_dataset_from_directory(datadir, batch_size= batch_size, image_size= image_size)
    data = data.map(lambda x,y: (x/255, y)) #normaliza los datos

    #cálculos de tamaños de los subsets de entrenamiento y validación
    train_size = int(len(data)*train_size)
    valid_size = int(len(data)*valid_size)

    #asignación de tamaños a cada subset
    train = data.take(train_size)
    valid = data.skip(train_size).take(valid_size)
    test = data.skip(train_size+valid_size).take(len(data)-(train_size+valid_size))

    return train, test, valid

#takes a model as input and returns the same model, but reinitialized
def reinitialize(model):
    json_string = model.to_json()
    return model_from_json(json_string)

#it takes the models, the output path, the train, test and valid subsets, 
# the number of epochs, the optimizer and if you want to use GPU as inputs,
# it saves the models in the output path and the history of each model in the same directory
def training(models, output_path, train, test, valid, epochs = 20, optimizer = 'SGD', GPU = False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for address in models:
        model = load_model(address)
        model_name = address.split(models + r'/')[1].split('.h5')[0]
        print(f'Training {model_name}...')

        model = reinitialize(model)
        
        model.compile(optimizer, loss = tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy', tf.keras.metrics.AUC()])
        
        if GPU:
            with tf.device('/device:GPU:0'):
                hist = model.fit(train,
                epochs = epochs,
                validation_data = valid,
                callbacks = [tensorboard_callback , anne],
                use_multiprocessing=True)
        else:
            hist = model.fit(train,
            epochs = epochs,
            validation_data = valid,
            callbacks = [tensorboard_callback , anne],
            use_multiprocessing=True)

        model_dir = os.path.join(path, model_name)
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model.save(model_dir + address.split(cnns_root + r'/')[1])
        np.save(f'{model_dir}/{address.split(cnns_root + r"/")[1].split(".h5")[0]}_epoch_history.npy', hist.history)

#check if a string is a valid directory, if not, it asks for a new one
def check_dir(string):
    while not os.path.isdir(string):
        string = input(f'{string} is not a valid directory\nTry Again: ')
    return string


#This part of the code is for metrics analysis, the code above is for training the models and getting the data
class Prediction_Analysis():
    
    def __init__(self, model_dir = None, folders = None, dataset_class = bool, image_height = 224, image_width = 224, transformers = None):       
        self.model_dir = model_dir
        self.folders = folders
        self.image_height = image_height
        self.image_width = image_width
        self.transformers = transformers

    @tf.function #this decorator allows the predictions to be much faster
    def tf_predict(self, model_list, img = None, weights = None): 
        #the model and weight's lists have to be in the same order. i.e: [model_1, model_2], [weight_1, weight_2] 
        y_gorrito = 0
        if weights is None:
            weights = [1] * len(model_list)
        for model, weight in zip(model_list, weights):
            y_gorrito += tf.cast(model(tf.expand_dims(img/255., 0)), dtype=tf.float32)*weight
        return y_gorrito / sum(weights)
        
    def hf_predict(self, classifiers, img):
        y_gorrito = 0
        for classifier in classifiers:
                classifier = classifier(img)                            
                for clase in classifier:
                    if clase['label'] == 'Patacon-True':
                        y_gorrito += clase["score"]
        return y_gorrito / len(classifiers)

    @tf.function #this function works with selected image file formats
    def preprocess(self, image, size, source):
        height, width = size
        image_dir = os.path.join(source, image)
        img_file = tf.io.read_file(image_dir)
        img_decode = tf.image.decode_image(img_file, channels=3, expand_animations = False)
        resize = tf.image.resize(img_decode,(height, width))
        return resize
    
    #returns the dictionary that serves as input to the calculateStats function
    def getPredictions(self, mode = 'tf'):
        results = {}
        ensemble = [(load_model(model), model) for model in self.model_dir] if mode == 'tf' else [(pipeline("image-classification", model= transformer), transformer) for transformer in self.transformers]
        
        for model, model_name in ensemble:
            history = []
            for pair in self.folders:
                folder, isPatacon = pair
                for image in os.listdir(folder):
                    try: #preprocess the image
                        resize = self.preprocess(image, (self.image_height, self.image_width), folder)
                    except: 
                        continue            
                    if mode == 'tf':
                        y_gorrito = float(self.tf_predict([model], resize))
                    elif mode == 'transformers':
                        y_gorrito = float(self.hf_predict([model], os.path.join(folder, image)))
                    history.append((isPatacon,y_gorrito))
            results[model_name] = history
        return results

def calculateStats(results_dictionary, threshold, output_path):
    if type(threshold) != list:
        threshold = [threshold]
    statsResults={}
    for key, value in results_dictionary.items():
        statsResults[key]={}
        for t in threshold:
            labels=[]
            predictions=[]
            statsResults[key][t]={}
            for tupla in value:
                labels.append(tupla[0])
                if tupla[1] >= t:
                    predictions.append(1)
                else:
                    predictions.append(0)
            
            confMatrix=tf.math.confusion_matrix(labels, predictions)
            confusion_matrix_array = np.array(confMatrix)
            
            sensibility = confusion_matrix_array[1][1]/(confusion_matrix_array[1][1]+confusion_matrix_array[1][0])
            precision = confusion_matrix_array[1][1]/(confusion_matrix_array[1][1]+confusion_matrix_array[0][1])
            especificidad=confusion_matrix_array[0][0]/(confusion_matrix_array[0][0]+confusion_matrix_array[0][1])
            false_negative_rate = confusion_matrix_array[1][0]/(confusion_matrix_array[1][0]+confusion_matrix_array[0][0])
            f1Score= 2*sensibility*precision/(sensibility+precision)
            
            statsResults[key][t]['matrix'] = confusion_matrix_array
            statsResults[key][t]['sensibility'] = sensibility
            statsResults[key][t]['precision'] = precision
            statsResults[key][t]['especificidad'] = especificidad
            statsResults[key][t]['f1Score'] = f1Score
            statsResults[key][t]['false_negative_rate'] = false_negative_rate
            ## Ploting
            cm_display = mt.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_array, display_labels = ["No-patacon", "Patacon"])
            cm_display.plot()
            plt.title(key+ ' Threshold= ' + str(t))
            final_path = os.path.join(output_path, f'{key}_{str(t)}.png')
            plt.savefig(final_path)
    return statsResults

def saveDictionary(dictionary, path,file_name='oc-ic.pickle'):
    file_path = os.path.join(path,file_name)
    with open(file_path, "wb") as file:
        pickle.dump(dictionary, file)

def loadDictionary(path,file_name='oc-ic.pickle'):
    file_path = os.path.join(path,file_name)
    with open(file_path, "rb") as file:
        return pickle.load(file)

def train_and_save():
    models_path, output_path, training_data, epochs, gpu = get_training_data().values()
    train, test, valid = dataset(training_data)
    anne = ReduceLROnPlateau(monitor='val_auc', factor=0.2, patience=5, verbose=1, min_lr=0.001)
    training(models_path, output_path, train, test, valid)

def analyze_data():
    data = get_analysis_data()
    if data["framework"] == 'tf':
        analysis = Prediction_Analysis(
            model_dir = data["model_dir"], 
            folders = data["folders"].replace(' ', '').split(','), 
            image_height = data["image_height"], 
            image_width = data["image_width"], 
            score_threshold = data["score_threshold"],
            dataset_class = data["dataset_class"],
            output_path = data["output_path"])
        predictions = analysis.getPredictions(mode= 'tf')
        results = calculateStats(predictions, data["score_threshold"])

    elif data["framework"] == 'transformers':
        analysis = Prediction_Analysis(
            transformers = data["transformers"].replace(' ', '').split(','), 
            folders = data["folders"].replace(' ', '').split(','), 
            image_height = data["image_height"], 
            image_width = data["image_width"], 
            score_threshold= data["score_threshold"],
            dataset_class = data["dataset_class"],
            output_path = data["output_path"])
        predictions = analysis.getPredictions(mode= 'hf')
        results = calculateStats(predictions, data["score_threshold"])
    
    print('Done!')

if __name__ == "__main__":
    choice = input('Type T/t for training and saving, and P/p for predicting analysis: ')
    if choice.lower() == 't':
        train_and_save()
    elif choice.lower() == 'p':
        analyze_data()
    input("Press Enter to close...")