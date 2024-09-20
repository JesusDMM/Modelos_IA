import pandas as pd
from scipy.stats import iqr

def Mediana_iqr(csvs, metricas):
    dfs = pd.DataFrame()
    Modelo = ['4', '5', '6']
    
    for metrica in metricas:
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
                [Modelo[id],metrica, mediana_entrenamiento, iqr_entrenamiento,
                mediana_validacion, iqr_validacion,
                mediana_test, iqr_test],
            ]

            df = pd.DataFrame(data, columns=['Modelo','metrica', 'mediana entrenamiento', 'iqr entrenamiento',
                                 'mediana validacion', 'iqr validacion',
                                 'mediana test', 'iqr test'])
            dfs = pd.concat([dfs, df], ignore_index=True)
    dfs.to_csv('Iqr y Mediana de modelos no pre entrenados general.csv', index=False)


datos_cp = ['/home/jesus/SugarCane/Pruebas_Estadisticas/EfficientNetV2B0/Graficas_EfficientNetV2B0/Data/Metricas_Efficientnet_cp.csv',
            '/home/jesus/SugarCane/Pruebas_Estadisticas/DenseNet121/Graficas_DenseNet121/Data/Metrics_CP_DenseNet121.csv',
            '/home/jesus/SugarCane/Pruebas_Estadisticas/ResNet101V2/Graficas_ResNet101V2/Data/Metrics_CP_ResNet101V2.csv']

datos_sp = ['/home/jesus/SugarCane/Pruebas_Estadisticas/EfficientNetV2B0/Graficas_EfficientNetV2B0/Data/Metricas_Efficientnet_sp.csv',
            '/home/jesus/SugarCane/Pruebas_Estadisticas/DenseNet121/Graficas_DenseNet121/Data/Metrics_SP_DenseNet121.csv',
            '/home/jesus/SugarCane/Pruebas_Estadisticas/ResNet101V2/Graficas_ResNet101V2/Data/Metrics_SP_ResNet101V2.csv']
metricas = ['accuracy', 'micro_avg_precision', 'micro_avg_recall', 'micro_avg_f1', 'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1']
Mediana_iqr(datos_sp, metricas)