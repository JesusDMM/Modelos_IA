import numpy as np
import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt

def test_normalidad_combinada(datos_csv, metricas):
    Nombre_metricas = ['Accuracy', 'Micro avg precision', 'Micro avg recall', 'Micro avg f1', 'Macro avg precision', 'Macro avg recall', 'Macro avg f1']

    for j, metrica in enumerate(metricas):
        num_modelos = len(datos_csv)
        fig, axs = plt.subplots(num_modelos, 3, figsize=(12, 4 * num_modelos))

        for idx, csv in enumerate(datos_csv):
            data_frame = pd.read_csv(csv)
            data_type = ['train', 'val', 'test']

            axs[idx, 0].set_ylabel(f'Modelo {idx + 1}', fontsize=16)

            for i, tipo in enumerate(data_type):
                resultado = data_frame[data_frame['data_type'] == tipo]
                datos = resultado[metrica]
                
                pg.qqplot(datos, dist='norm', ax=axs[idx, i])
                axs[idx, i].set_title(f'Model {idx + 1} {tipo.capitalize()}')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        plt.suptitle(f'{Nombre_metricas[j]}', fontsize=20)

        plt.savefig(f'{Nombre_metricas[j]}.png')

        plt.close()

datos = [
    '/home/jesus/SugarCane/Pruebas_Estadisticas/EfficientNetV2B0/Graficas_EfficientNetV2B0/Data/Metricas_Efficientnet_cp.csv',
    '/home/jesus/SugarCane/Pruebas_Estadisticas/DenseNet121/Graficas_DenseNet121/Data/Metrics_CP_DenseNet121.csv',
    '/home/jesus/SugarCane/Pruebas_Estadisticas/ResNet101V2/Graficas_ResNet101V2/Data/Metrics_CP_ResNet101V2.csv',
    '/home/jesus/SugarCane/Pruebas_Estadisticas/EfficientNetV2B0/Graficas_EfficientNetV2B0/Data/Metricas_Efficientnet_sp.csv',
    '/home/jesus/SugarCane/Pruebas_Estadisticas/DenseNet121/Graficas_DenseNet121/Data/Metrics_SP_DenseNet121.csv',
    '/home/jesus/SugarCane/Pruebas_Estadisticas/ResNet101V2/Graficas_ResNet101V2/Data/Metrics_SP_ResNet101V2.csv'
]
metricas = ['accuracy', 'micro_avg_precision', 'micro_avg_recall', 'micro_avg_f1', 'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1']
test_normalidad_combinada(datos, metricas)
