import os
import cv2
import pickle
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
from math import floor
from sklearn import metrics as mt
from keras.models import model_from_json
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def get_training_data(): #this function asks the user all the data needed for training
    
    sfs = { 
    'non_trained_models_path' : input("Type the directory where the non-trained models are stored (if there are none, type N/n): "),
    'trained_models_path' : input("Type the directory where the models to be fine-tuned are stored (if there are none, type N/n):"),
    'output_path' : input("Type the directory where you want to save the models and the history of each model: "),
    'dataset' : input("Type the directory where the dataset is stored (it has to be separated into classes, and not splitted in train, test or valid subsets): "),
    'epochs' : int(input("Type the number of epochs you want to train the models: ")),
    'models_folders' : {
        0 : None,
        1 : None},
    'models_paths' : []
    }
    
    aux = []
    
    for label, value in sfs.items():
        
        if label == 'non_trained_models_path' and sfs[label].lower() != "n":
            aux.extend(os.listdir(value))
            sfs['models_folders'][0] = value
            
        elif label == 'trained_models_path' and sfs[label].lower() != "n":
            aux.extend(os.listdir(value))
            sfs['models_folders'][1] = value           
        
        if label in ['output_path', 'dataset']:
            #check if the values are in a valid address directory
            check_dir(value)
            sfs[label] = check_dir(value)
            
    for model in aux:
        if model in os.listdir(sfs["models_folders"][0]):
            sfs["models_paths"].append((model, 0))
        elif model in os.listdir(sfs["models_folders"][1]):
            sfs["models_paths"].append((model, 1))
        
    
    return sfs

def get_analysis_data(): #same as get_training_data() but for analysis
    framework = input("Type the framework you used to train the models (tf or hf): ") #tensorflow/huggingface
    sfs = {}

    if framework.lower() == 'tf':
        sfs = {
        "framework" : "tf",
        "positive_class_folders" : input("Type the directories where the folders with the positive images are stored separated by commas (if there are none, type N/n): "),
        "negative_class_folders" : input("Type the directories where the folders with the negative images are stored separated by commas (if there are none, type N/n): "),
        
        'folders_classes' : {
        0 : None,
        1 : None},
            
        "folders" : [],
        
        "models_path" : input("Type the directory where the models are stored: "),
        "image_height" : input("Type the height of the images: "),
        "image_width" : input("Type the width of the images: "),
        "output_path" : input("Type the directory where you want to save the results: "),
        "score_threshold" : [i/100 for i in range(1, 100, 3)]}
        
        aux = []

    elif framework.lower() == 'hf':
        sfs = {
            "framework" : "hf",
            "transformers names" : input("Type the names of the transformers you want to use, separated by commas: "),
            "positive_class_folders" : input("Type the directories where the folders with the positive images are stored separated by commas (if there are none, type N/n): "),
            "negative_class_folders" : input("Type the directories where the folders with the negative images are stored separated by commas (if there are none, type N/n): "),
        
            'folders_classes' : {
            0 : None,
            1 : None},
            
            "folders" : [],
            "image_height" : input("Type the desired height of the images: "),
            "image_width" : input("Type the desired width of the images: "),
            "output_path" : input("Type the directory where you want to save the results: "),
            "score_threshold" : [i/100 for i in range(1, 100, 3)]}
        
    for label, value in sfs.items():

    if label == "negative_class_folders" and sfs[label].lower() != "n":
        addresses = value.replace(" ", "").split(",")

        for address in addresses:
            sfs["folders"].append((address, 0))

    elif label == "positive_class_folders" and sfs[label].lower() != "n":
        addresses = value.replace(" ", "").split(",")

        for address in addresses:
            sfs["folders"].append((address, 1))
            
    return sfs


#it takes the path to the dataset, the batch size, the image size, the train size and
#the valid size as inputs, and returns the train, test and valid subsets
def dataset(datadir, batch_size = 16, image_size = (224, 224), train_size = .6, valid_size = .2):
    AUTOTUNE = tf.data.AUTOTUNE
    data =  image_dataset_from_directory(datadir, batch_size= batch_size, image_size= image_size, shuffle=True)
    data = data.map(lambda x,y: (x/255, y)) #normaliza los datos

    #cálculos de tamaños de los subsets de entrenamiento y validación
    train_size = int(len(data)*train_size)
    valid_size = int(len(data)*valid_size)

    #asignación de tamaños a cada subset
    train = data.take(train_size).prefetch(buffer_size=AUTOTUNE)
    valid = data.skip(train_size).take(valid_size).prefetch(buffer_size=AUTOTUNE)
    test = data.skip(train_size+valid_size).take(len(data)-(train_size+valid_size)).prefetch(buffer_size=AUTOTUNE)
    

    return train, test, valid

#takes a model as input and returns the same model, but reinitialized with random weights and a data augmentation layer
def reinitialize(model, optimizer, finetune=False, pool="max", fine_tune_at=None):
    
    data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),])

    inputs = tf.keras.Input(shape = (224, 224, 3))
    x = data_augmentation(inputs)
    
    if finetune:
        for layer in model.layers[:fine_tune_at]: #freeze layer from a number onwards
            layer.trainable = False
        x = model(x)
        
        if len(x.shape) != 4:
            pass
        elif pool == "avg":
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif pool == "max":
            x = tf.keras.layers.MaxPooling2D()(x) 
            x = tf.keras.layers.Flatten()(x) #due to possible mismatch of shapes
        outputs = tf.keras.layers.Dense(1, activation = "sigmoid")(x)
            
    else:
        json_string = model.to_json()
        json_model = model_from_json(json_string)
        outputs = json_model(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy', tf.keras.metrics.AUC()])
        
    return model
#it takes the models, the output path, the train, test and valid subsets,
# the number of epochs and the optimizers 
# it saves the models in the output path and the history of each model in the same directory
def training(models_names, folders, output_path, train, test, valid, epochs, optimizers = ['SGD', 'adam'] ):
    
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() 
    #non_trained_addresses = [os.path.join(models, model_name) for model_name in os.listdir(models)]
    #trained_addresses = []
    
    cnns_root = "ptctrn_v"
    anne = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, verbose=1, min_lr=5e-5)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)        
        
    def execute(addresses, optimizers, output_path):
        
        for model_file, index in addresses:
            folder = folders[index]
            model = load_model(os.path.join(folder, model_file))
            model_name = model_file.split(".h5")[0]
            print(f'Training {model_name}...')
            
            for optimizer in optimizers:
                model = reinitialize(model, optimizer) if index == 0 else reinitialize(model, optimizer, finetune = True, 
                                                                                       fine_tune_at = int(floor(0.4*len(model.layers))))
                hist = model.fit(train,
                                 epochs=epochs,
                                 validation_data=valid,
                                 callbacks=[anne],
                                 use_multiprocessing=True)
                
                model_dir = os.path.join(output_path, f"{model_name}-{optimizer}")
                
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                    
                model.save(os.path.join(model_dir, model_file)) #address.split(cnns_root + r'/')[1]
                #np.save(f'{model_dir}/{address.split(cnns_root + r"/")[1].split(".h5")[0]}_epoch_history.npy', hist.history)
                np.save(os.path.join(model_dir, model_name+"_epoch_history.npy"), hist.history)
        
    with strategy.scope():
         execute(models_names, optimizers, output_path)

#check if a string is a valid directory, if not, it asks for a new one
def check_dir(string):
    while not os.path.exists(string):
        try:
            os.makedirs(string)
        except:
            string = input(f'{string} is not a valid directory\nTry Again: ')
    return string


#This part of the code is for metrics analysis, the code above is for training the models and getting the data
class Prediction_Analysis():

    def __init__(self, model_dir = None, folders = None, image_height = 224, image_width = 224, transformers = None):
        self.model_dir = model_dir
        self.folders = folders
        self.image_height = image_height
        self.image_width = image_width
        self.transformers = transformers

    #@tf.function #this decorator allows the predictions to be much faster
    def old_tf_predict(self, model_list, img = None, weights = None):
        #the model and weight's lists have to be in the same order. i.e: [model_1, model_2], [weight_1, weight_2]
        img_float = tf.cast(img, tf.float32)
        y_gorrito = 0
        if weights is None:
            weights = [1] * len(model_list)
        for model, weight in zip(model_list, weights):
            y_gorrito += tf.cast(model(tf.expand_dims(img_float/255., 0)), dtype=tf.float32)*weight
        
        return y_gorrito / sum(weights)
    
    def tf_predict(self, model_list, img = None, weights = None):
        #the model and weight's lists have to be in the same order. i.e: [model_1, model_2], [weight_1, weight_2]
        img_float = tf.cast(img, tf.float32)
        y_gorrito = 0
        if weights is None:
            weights = [1] * len(model_list)
        for model, weight in zip(model_list, weights):
            y_gorrito += model.predict(tf.expand_dims(img_float/255., 0))
        return y_gorrito / sum(weights)

    def hf_predict(self, classifiers, img, label = 'Patacon-True'): #gets the "Patacón-True" label probability
        y_gorrito = 0
        for classifier in classifiers:
                classifier = classifier(img)
                for clase in classifier:
                    if clase['label'] == label:
                        y_gorrito += clase["score"]
        return y_gorrito / len(classifiers)

    @tf.function #this function works with selected image file formats
    def preprocess(self, image, size, source):
        height, width = int(size[0]), int(size[1])
        size_tensor = tf.constant([height, width], dtype=tf.int32)
        image_dir = os.path.join(source, image)
        img_file = tf.io.read_file(image_dir)
        img_decode = tf.image.decode_image(img_file, channels=3, expand_animations = False)
        resize = tf.image.resize(img_decode, size_tensor)
        return resize
    
    def generic_preprocess(self, image, size, source):
        size_tuple = (int(size[0]), int(size[1]))
        image_dir = os.path.join(source, image)
        img = cv2.imread(image_dir)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize = cv2.resize(img_rgb, size_tuple)
        return resize

    #returns the dictionary that serves as input to the calculateStats function
    def getPredictions(self, mode = 'tf'):
        
        results = {}
        print("Loading models...")
        ensemble = [(load_model(os.path.join(self.model_dir, model, model.split("-")[0]+".h5")), model.split("-")[0], model.split("-")[1]) for model in os.listdir(self.model_dir)] if mode == 'tf' else [(pipeline("image-classification", model= transformer), transformer) for transformer in self.transformers]
        print("Loaded!")
        for model, model_name, optimizer in ensemble:
            print(f"Performing inference with: {model_name} using {optimizer} as optimizer...")
            history = []
            for pair in self.folders:
                folder, isPatacon = pair
                print(pair)
                for image in os.listdir(folder):
                    try: #preprocess the image
                        resize = self.preprocess(image, (self.image_height, self.image_width), folder)
                    except:
                        resize = self.generic_preprocess(image, (self.image_height, self.image_width), folder)
                    #    continue
                    if mode == 'tf':
                        y_gorrito = float(self.tf_predict([model], resize))
                        print(y_gorrito)
                    elif mode == 'transformers':
                        y_gorrito = float(self.hf_predict([model], os.path.join(folder, image)))
                    history.append((isPatacon,y_gorrito))
            results[model_name+"-"+optimizer] = history
        saveDictionary(results, path = "/kaggle/working")
        return results

def graphAnalysis(dic = None, stat_name='ROC', allInOne=True, save = True, path = os.getcwd()): 
    #part of the calculateStats function, initializes and plot all variables needed
    stat_names = ['ROC', 'SENSIBILITY', 'RECALL', 'R','P', 'PRECISION','F1', 'FNR']
    choose='ROC'
    line_styles = ['-','--','-.',':']
    plt.figure(figsize=(12.8,9.6))
    plt.grid()
    if stat_name in stat_names:
        choose=stat_name
    for model_name, history in dic.items():
        sensibility = []
        precision = []
        especificidad = []
        thresholds = []
        fnr = []
        f1 = []
        for t,value in history.items():
            sensibility.append(value['sensibility'])
            precision.append(value['precision'] if value['precision']==value['precision'] else 0)
            thresholds.append(t)
            especificidad.append(1-value['especificidad'])
            f1.append(2*value['sensibility']*value['precision']/(value['sensibility']+value['precision']))
            fnr.append(value['false_negative_rate'])
        xname='Threshold'
        if choose in ['SENSIBILITY', 'RECALL', 'R']:
            x=thresholds
            y=sensibility
            yname='Sensibilidad'
        elif choose in ['P', 'PRECISION']:
            x=thresholds
            y=precision
            yname='Precision'
        elif choose == 'F1':
            x=thresholds
            y=f1
            yname='F1-Score'
        elif choose == 'FNR':
            x = thresholds
            y = fnr
            yname='FNR'
        else: # ROC by default
            x=especificidad
            y=sensibility
            xname='Especificidad'
            yname='Sensibilidad'
        plt.plot(x, y, label=model_name, linestyle =line_styles[random.randint(0,3)])
        if not allInOne:
            plt.title("Curva "+stat_name)
            plt.legend(fontsize=6)
            plt.xlabel(xname)
            plt.ylabel(yname)
            plt.show()
            if save:
                plt.savefig(os.path.join(path, model_name, stat_name))
    if allInOne:
        plt.title("Curva "+stat_name)
        plt.legend(fontsize=12)
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.show()
        if save:
            plt.savefig(os.path.join(path, stat_name))

def graphAll(dic, path): #graph all the options in graphAnalysis
    for stat_name in ['ROC', 'RECALL', 'PRECISION','F1', 'FNR']:
        graphAnalysis(dic = dic, stat_name = stat_name, path = path)

def calculateStats(results_dictionary, threshold, output_path):
    if type(threshold) != list:
        threshold = [threshold] #hehe
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
            confusion_matrix_array = np.array(confMatrix) #for the following operations below:

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

#a function that executes all functions needed to perform and save training
def train_and_save():
    non_trained_dir, trained_dir, output_path, training_data, epochs, folders, models_paths = get_training_data().values()
    train, test, valid = dataset(training_data)
    anne = ReduceLROnPlateau(monitor='val_auc', factor=0.2, patience=5, verbose=1, min_lr=0.001)
    training(models_paths, folders, output_path, train, test, valid, epochs)

#a function that executes all functions needed to perform testing and to save its results
def analyze_data(plot = False, save = True):
    data = get_analysis_data()
    if data["framework"] == 'tf': #tensorflow
        analysis = Prediction_Analysis(
            model_dir = data["models_path"],
            folders = data["folders"],
            image_height = data["image_height"],
            image_width = data["image_width"])
        
        predictions = analysis.getPredictions(mode= 'tf')
        results = calculateStats(predictions, data["score_threshold"])

        if plot and len(data["score_threshold"]) > 1:
            graphAll(results, data["output_path"])
        elif plot and len(data["score_threshold"]) <= 1:
            print('You need more than one threshold to create a graph')

    elif data["framework"] == 'transformers': #huggingface
        analysis = Prediction_Analysis(
            transformers = data["transformers"].replace(' ', '').split(','),
            folders = data["folders"],
            image_height = data["image_height"],
            image_width = data["image_width"],
            score_threshold= data["score_threshold"])
        
        predictions = analysis.getPredictions(mode= 'hf')
        results = calculateStats(predictions, data["score_threshold"])

        if plot and data["score_threshold"] > 1:
            graphAll(results, data["output_path"])
        elif plot and data["score_threshold"] <= 1:
            print('You need more than one threshold to create a graph') #one coordinate plotted is just a point
    print('Done!')

if __name__ == "__main__":
    while True:
        choice = input('Type T/t for training and saving, and P/p for predicting analysis: ')
    
        if choice.lower() == 't':
            train_and_save()
        elif choice.lower() == 'p':
            analyze_data()
            
        choice = input("If you wish to execute a new task, press Y/y.\nOtherwise, press any key: ")
        
        if choice.lower == "c":
            continue
        else:
            break
