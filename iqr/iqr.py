import pandas as pd
from scipy.stats import iqr

def Mediana_iqr(csvs):
    dfs = pd.DataFrame()
    metrica = 'accuracy'
    Modelo = ['EfficientNetV2B0', 'DenseNet121', 'ResNet101V2']
    
    for id, csv in enumerate(csvs):
        datos = pd.read_csv(csv)
        datos_entrenamiento = datos[datos['data_type'] == 'train'][metrica]
        datos_validacion = datos[datos['data_type'] == 'val'][metrica]
        datos_test = datos[datos['data_type'] == 'test'][metrica]
        
        mediana_entrenamiento = datos_entrenamiento.median()
        mediana_validacion = datos_validacion.median()
        mediana_test = datos_test.median()
        
        iqr_entrenamiento = iqr(datos_entrenamiento)
        iqr_validacion = iqr(datos_validacion)
        iqr_test = iqr(datos_test)
                
        data = [
            [metrica, Modelo[id], iqr_entrenamiento, mediana_entrenamiento,
                                  iqr_validacion, mediana_validacion,
                                  iqr_test, mediana_test],
        ]
        df = pd.DataFrame(data, columns=['Metrica', 'Modelo', 'iqr entrenamiento', 'mediana entrenamiento',
                                                              'iqr validacion', 'mediana validacion',
                                                              'iqr test', 'mediana test'])
        dfs = pd.concat([dfs, df], ignore_index=True)
    dfs.to_csv('Iqr y Mediana de modelos no pre entrenados.csv', index=False)


datos_cp = ['/home/jesus/SugarCane/Pruebas_Estadisticas/EfficientNetV2B0/Graficas_EfficientNetV2B0/Data/Metricas_Efficientnet_cp.csv',
            '/home/jesus/SugarCane/Pruebas_Estadisticas/DenseNet121/Graficas_DenseNet121/Data/Metrics_CP_DenseNet121.csv',
            '/home/jesus/SugarCane/Pruebas_Estadisticas/ResNet101V2/Graficas_ResNet101V2/Data/Metrics_CP_ResNet101V2.csv']

datos_sp = ['/home/jesus/SugarCane/Pruebas_Estadisticas/EfficientNetV2B0/Graficas_EfficientNetV2B0/Data/Metricas_Efficientnet_sp.csv',
            '/home/jesus/SugarCane/Pruebas_Estadisticas/DenseNet121/Graficas_DenseNet121/Data/Metrics_SP_DenseNet121.csv',
            '/home/jesus/SugarCane/Pruebas_Estadisticas/ResNet101V2/Graficas_ResNet101V2/Data/Metrics_SP_ResNet101V2.csv']

Mediana_iqr(datos_sp)