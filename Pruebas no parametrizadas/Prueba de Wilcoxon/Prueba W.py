from scipy import stats
import pandas as pd

def Kruskal_wallis(metricas, csvs, nombre_csv, tipo):
    dfs = pd.DataFrame()
    
    for id, csv in enumerate(csvs):
        datos = pd.read_csv(csv)
        for metrica in metricas:
            datos_validacion = datos[datos['data_type'] == 'train'][metrica].sort_values(ascending=True)
            datos_test = datos[datos['data_type'] == 'test'][metrica].sort_values(ascending=True)
            
            stat, p_value = stats.wilcoxon(datos_validacion, datos_test)
            #H0: las distribuciones entre val y test son iguales
            #H1: las distribuciones entre val y test son diferentes
            h = 'h0' if p_value > 0.05 else 'h1'
                
            data = [
                [metrica, tipo[id], stat, p_value, h],
            ]
            df = pd.DataFrame(data, columns=['Metrica', 'Modelo', 'Stat', 'p_value', 'hipotesis'])
            dfs = pd.concat([dfs, df], ignore_index=True)
    dfs.to_csv(nombre_csv, index=False)


datos_cp = ['/home/jesus/SugarCane/Pruebas_Estadisticas/EfficientNetV2B0/Graficas_EfficientNetV2B0/Data/Metricas_Efficientnet_cp.csv',
            '/home/jesus/SugarCane/Pruebas_Estadisticas/DenseNet121/Graficas_DenseNet121/Data/Metrics_CP_DenseNet121.csv',
            '/home/jesus/SugarCane/Pruebas_Estadisticas/ResNet101V2/Graficas_ResNet101V2/Data/Metrics_CP_ResNet101V2.csv']

datos_sp = ['/home/jesus/SugarCane/Pruebas_Estadisticas/EfficientNetV2B0/Graficas_EfficientNetV2B0/Data/Metricas_Efficientnet_sp.csv',
            '/home/jesus/SugarCane/Pruebas_Estadisticas/DenseNet121/Graficas_DenseNet121/Data/Metrics_SP_DenseNet121.csv',
            '/home/jesus/SugarCane/Pruebas_Estadisticas/ResNet101V2/Graficas_ResNet101V2/Data/Metrics_SP_ResNet101V2.csv']

metricas = ['accuracy', 'micro_avg_precision', 'micro_avg_recall', 'micro_avg_f1', 'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1']
nombre_csv = 'Wilcoxon train test sp.csv'
tipo = ['EfficientNetV2B0', 'DenseNet121', 'ResNet101V2']
Kruskal_wallis(metricas, datos_sp, nombre_csv, tipo)