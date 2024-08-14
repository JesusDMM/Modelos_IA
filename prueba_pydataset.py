import tensorflow as tf

import numpy as np
import pandas as pd

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from obtener_info import *
#from cargar_imagenes import *
from generator import *

def DenseNet121(numero_clases, entrada, pesos):
    modelo = models.Sequential()
    modelo_densenet121 = tf.keras.applications.DenseNet121(
        include_top=False,
        weights=pesos,
        input_shape=entrada,
        pooling='max'
    )
    modelo.add(modelo_densenet121)
    modelo.add(Dense(680, kernel_regularizer=l2(0.01) ,activation='relu'))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(units=numero_clases, activation='softmax'))
    modelo.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return modelo

def obtener_identificar_clase(image_df):
    class2id = dict()
    for idx, clase in enumerate(pd.DataFrame(image_df['Label'].unique(), columns=['Label'])['Label']):
        class2id[clase] = idx
    labels_array = np.array(list(class2id.values()))
    return class2id, labels_array

def sacar_imagenes_generales(imagenes_generales, modelo):
    predictions = []
    etiquetas = []

    dataset_iterator = iter(imagenes_generales)

    while True:
        try:
            images, labels = next(dataset_iterator)
            batch_predictions = np.argmax(modelo.predict(images), axis=1)
            batch_labels = np.argmax(labels, axis=1)
            predictions.extend(batch_predictions)
            etiquetas.extend(batch_labels)
        except StopIteration:
            break
    predictions = np.array(predictions)
    etiquetas = np.array(etiquetas)
    del(images, labels, batch_predictions, batch_labels)
    return etiquetas, predictions

def obtener_metricas_micro (etiquetas, predictions, tipo, labels_array):
    avg_precision = precision_score(etiquetas, predictions, labels=labels_array,average=tipo)
    avg_recall = recall_score(etiquetas, predictions, labels=labels_array,average=tipo)
    avg_f1 = f1_score(etiquetas, predictions, labels=labels_array,average=tipo)
    return avg_precision, avg_recall, avg_f1

results_df = pd.DataFrame({})
classification_metrics = pd.DataFrame(columns=['accuracy', 
                                                'micro_avg_precision',          
                                                'micro_avg_recall',
                                                'micro_avg_f1',
                                                'macro_avg_precision',
                                                'macro_avg_recall',
                                                'macro_avg_f1',
                                                'iteration',
                                                'fold',
                                                'data_type'])

if __name__ == "__main__":
    
    pesos = None
    image_df = ruta_df()
    train_df, test_df = obtener_train_test(image_df)
    label_map, labels = obtener_identificar_clase(image_df)
    numero_clases = 5
    
    # Parameters
    params = {'batch_size': 32,
            'dim': (493,493),
            'n_channels': 3,
            'n_classes': 5,
            'shuffle': True}

    K = 5
    kf = StratifiedKFold(n_splits=K, shuffle=True)

    for iteration in np.arange(1, 2):
        fold = 1
        
        for train_indices, test_indices in kf.split(train_df, train_df['Label']):
            
            print(f'Iteracion {iteration} Fold {fold} ')
            
            datos_entranamiento = train_df.iloc[train_indices]
            datos_validacion = train_df.iloc[test_indices]
            
            diccionario_rutas = {}
            
            ruta_entrenamiento = datos_entranamiento['Filepath'].tolist()
            ruta_validacion = datos_validacion['Filepath'].tolist()
            ruta_test = test_df['Filepath'].tolist()
            
            #diccionario que almacena las rutas
            diccionario_rutas['entrenamiento'] = ruta_entrenamiento
            diccionario_rutas['validacion'] = ruta_validacion
            diccionario_rutas['test'] = ruta_test
            
            label_entrenamiento = datos_entranamiento['Label']
            label_validacion = datos_validacion['Label']
            label_test = test_df['Label']
            
            label_entrenamiento_numerica_onehot = [label_map[label] for label in label_entrenamiento]
            label_validacion_numerica_onehot = [label_map[label] for label in label_validacion]
            label_test_numerica_onehot = [label_map[label] for label in label_test]
            
            #label_entrenamiento_categorical = to_categorical(label_entrenamiento_numerica_onehot, num_classes=numero_clases)
            #label_validacion_categorical = to_categorical(label_validacion_numerica_onehot, num_classes=numero_clases)
            #label_test_categorical = to_categorical(label_test_numerica_onehot, num_classes=numero_clases)
            
            diccionario_label = {}
            
            for id, ruta in enumerate(diccionario_rutas['entrenamiento']):
                diccionario_label[ruta] = label_entrenamiento_numerica_onehot[id]

            for id, ruta in enumerate(diccionario_rutas['validacion']):
                diccionario_label[ruta] = label_validacion_numerica_onehot[id]

            for id, ruta in enumerate(diccionario_rutas['test']):
                diccionario_label[ruta] = label_test_numerica_onehot[id]
                
            #training_generator = DataGenerator(partition['train'], labels, **params)
            #validation_generator = DataGenerator(partition['validation'], labels, **params)
            
            imagenes_entramiento = gen(diccionario_rutas['entrenamiento'],labels=diccionario_label,**params)
            imagenes_validacion = gen(diccionario_rutas['validacion'], diccionario_label, **params)
            imagenes_test = gen(diccionario_rutas['test'], diccionario_label, **params)
            
            
            Epocas = 10
            modelo = DenseNet121(5, (493, 493, 3), pesos)
            registro_general = modelo.fit(imagenes_entramiento, 
                                          epochs=Epocas, 
                                          validation_data=imagenes_validacion, 
                                          verbose=0)
            print('fin')
            break
            '''
            
            iteration_history = pd.DataFrame(registro_general.history)
            iteration_history['epoch'] = np.arange(Epocas) + 1
            iteration_history['iteration'] = iteration
            iteration_history['fold'] = fold 
            
            entrenamiento_predictions = []
            entrenamiento_etiquetas = []
            
            validacion_predictions = []
            validacion_etiquetas = []
            
            test_predictions = []
            test_etiquetas = []
            
            entrenamiento_etiquetas, entrenamiento_predictions = sacar_imagenes_generales (imagenes_entramiento, modelo)
            validacion_etiquetas, validacion_predictions = sacar_imagenes_generales(imagenes_validacion, modelo)
            test_etiquetas, test_predictions = sacar_imagenes_generales(imagenes_test, modelo)
            
            del(imagenes_entramiento, imagenes_validacion, imagenes_test)
            
            train_accuracy = accuracy_score(entrenamiento_etiquetas, entrenamiento_predictions)
            val_accuracy = accuracy_score(validacion_etiquetas, validacion_predictions)
            test_accuracy = accuracy_score(test_etiquetas, test_predictions)
            
            
            train_micro_avg_precision, train_micro_avg_recall, train_micro_avg_f1 = obtener_metricas_micro(entrenamiento_etiquetas, entrenamiento_predictions, 'micro', labels)
            
            val_micro_avg_precision, val_micro_avg_recall, val_micro_avg_f1 = obtener_metricas_micro(validacion_etiquetas, validacion_predictions, 'micro', labels)
            
            test_micro_avg_precision, test_micro_avg_recall, test_micro_avg_f1 = obtener_metricas_micro(test_etiquetas, test_predictions, 'micro', labels)
            
            train_macro_avg_precision, train_macro_avg_recall, train_macro_avg_f1 = obtener_metricas_micro(entrenamiento_etiquetas, entrenamiento_predictions, 'macro', labels)
            
            val_macro_avg_precision, val_macro_avg_recall, val_macro_avg_f1 = obtener_metricas_micro(validacion_etiquetas, validacion_predictions, 'macro', labels)
            
            test_macro_avg_precision, test_macro_avg_recall, test_macro_avg_f1 = obtener_metricas_micro(test_etiquetas, test_predictions, 'macro', labels)

            iteration_metrics = pd.DataFrame({'accuracy' : [train_accuracy, val_accuracy, test_accuracy],
                                                    'micro_avg_precision' : [train_micro_avg_precision, val_micro_avg_precision, test_micro_avg_precision],
                                                    'micro_avg_recall' : [train_micro_avg_recall, val_micro_avg_recall, test_micro_avg_recall],
                                                    'micro_avg_f1' : [train_micro_avg_f1, val_micro_avg_f1, test_micro_avg_f1],
                                                    'macro_avg_precision' : [train_macro_avg_precision, val_macro_avg_precision, test_macro_avg_precision],
                                                    'macro_avg_recall' : [train_macro_avg_recall, val_macro_avg_recall, test_macro_avg_recall],
                                                    'macro_avg_f1' : [train_macro_avg_f1, val_macro_avg_f1, test_macro_avg_f1],
                                                    'iteration' : [iteration, iteration, iteration],
                                                    'fold' : [fold, fold, fold],
                                                    'data_type' : ['train', 'val', 'test']})
            
            
            results_df = pd.concat([results_df, iteration_history], ignore_index=True)
            classification_metrics = pd.concat([classification_metrics, iteration_metrics], ignore_index=True)
            fold += 1
        results_df.to_csv('/home/Pruebas_Exhaustivas/DenseNet121/cross_validation_history_' + 'DenseNet121_B_SP.csv', index=False)
        classification_metrics.to_csv('/home/Pruebas_Exhaustivas/DenseNet121/cross_validation_metrics_' + 'DenseNet121_B_SP.csv', index=False)
        '''