import tensorflow as tf

import numpy as np
import pandas as pd

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import models
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from obtener_info import *

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
    return labels_array

def sacar_imagenes_generales (imagenes_generales, modelo):
    
    predictions = []
    etiquetas = []
    
    for i in range(len(imagenes_generales)):
            
        images, labels = next(imagenes_generales)

        batch_predictions = np.argmax(modelo.predict(images), axis=1)
        batch_labels =  np.argmax(labels, axis=1)
            
        predictions.extend(batch_predictions)
        etiquetas.extend(batch_labels)
        
    predictions = np.array(predictions)
    etiquetas = np.array(etiquetas)
    
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
    
    pesos = 'imagenet'
    input =  tf.keras.applications.densenet.preprocess_input
    image_df = ruta_df()
    labels = obtener_identificar_clase(image_df)
    train_df, test_df = obtener_train_test(image_df)
    generator = input_correspondiente(input)
    
    K = 5
    kf = StratifiedKFold(n_splits=K, shuffle=True)

    for iteration in np.arange(31) + 1:
        fold = 1
        
        for train_indices, test_indices in kf.split(train_df, train_df['Label']):
            
            print(f'Iteracion {iteration} Fold {fold} ')
            
            datos_entranamiento = train_df.iloc[train_indices]
            datos_validacion = train_df.iloc[test_indices]
            
            imagenes_entramiento = obtener_datos_entrenar(generator, datos_entranamiento)
            imagenes_validacion = obtener_datos_entrenar(generator, datos_validacion)
            imagenes_test = obtener_datos_entrenar(generator, test_df)
            
            Epocas = 20
            modelo = DenseNet121(5, (493, 493, 3), pesos)
            registro_general = modelo.fit(imagenes_entramiento, epochs=Epocas, validation_data=imagenes_validacion, verbose=0)
            
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
        results_df.to_csv('/home/Pruebas_Exhaustivas/DenseNet121/cross_validation_history_' + 'DenseNet121_BCP.csv', index=False)
        classification_metrics.to_csv('/home/Pruebas_Exhaustivas/DenseNet121/cross_validation_metrics_' + 'DenseNet121_BCP.csv', index=False)