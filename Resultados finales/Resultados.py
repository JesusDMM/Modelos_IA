from scipy import stats
import pandas as pd

def obtener_metricas(datos):
    media = datos.mean()
    desviacion_estandar = datos.std()
    mediana = datos.median()
    return media, desviacion_estandar, mediana

def Kruskal_wallis(metricas, csvs, nombre_csv, tipo):
    dfs = pd.DataFrame()
    
    for id, csv in enumerate(csvs):
        datos = pd.read_csv(csv)
        for metrica in metricas:
            
            datos_entrenamiento = datos[datos['data_type'] == 'train'][metrica]
            datos_validacion = datos[datos['data_type'] == 'val'][metrica]
            datos_test = datos[datos['data_type'] == 'test'][metrica]
            
            media_train, desv_train, med_train = obtener_metricas(datos_entrenamiento)
            media_val, desv_val, med_val = obtener_metricas(datos_validacion)
            media_test, desv_test, med_test = obtener_metricas(datos_test)     
            
                
            data = [
                [metrica, tipo[id],media_train, desv_train, med_train,media_val, desv_val, med_val,media_test, desv_test, med_test]
            ]
            df = pd.DataFrame(data, columns=['Metrica', 'Modelo',
                                             'media_train', 'desv_train', 'med_train',
                                             'media_val', 'desv_val', 'med_val', 
                                             'media_test', 'desv_test', 'med_test'])
            dfs = pd.concat([dfs, df], ignore_index=True)
    dfs.to_csv(nombre_csv, index=False)


datos_cp = ['/home/jesus/SugarCane/Pruebas_Estadisticas/EfficientNetV2B0/Graficas_EfficientNetV2B0/Data/Metricas_Efficientnet_cp.csv',
            '/home/jesus/SugarCane/Pruebas_Estadisticas/DenseNet121/Graficas_DenseNet121/Data/Metrics_CP_DenseNet121.csv',
            '/home/jesus/SugarCane/Pruebas_Estadisticas/ResNet101V2/Graficas_ResNet101V2/Data/Metrics_CP_ResNet101V2.csv']

datos_sp = ['/home/jesus/SugarCane/Pruebas_Estadisticas/EfficientNetV2B0/Graficas_EfficientNetV2B0/Data/Metricas_Efficientnet_sp.csv',
            '/home/jesus/SugarCane/Pruebas_Estadisticas/DenseNet121/Graficas_DenseNet121/Data/Metrics_SP_DenseNet121.csv',
            '/home/jesus/SugarCane/Pruebas_Estadisticas/ResNet101V2/Graficas_ResNet101V2/Data/Metrics_SP_ResNet101V2.csv']

metricas = ['accuracy', 'micro_avg_precision', 'micro_avg_recall', 'micro_avg_f1', 'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1']
nombre_csv = 'Medidas de tendencia central generales (None).csv'
tipo = ['EfficientNetV2B0', 'DenseNet121', 'ResNet101V2']
Kruskal_wallis(metricas, datos_sp, nombre_csv, tipo)