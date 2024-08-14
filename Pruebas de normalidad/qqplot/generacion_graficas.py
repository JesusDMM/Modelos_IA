import numpy as np
import pingouin as pg
import pandas as pd 
import matplotlib.pyplot as plt

def test_normalidad(data_frame, metricas, data_type):
    for metrica in metricas:
        fig, axs = plt.subplots(1, len(data_type), figsize=(4 * len(data_type), 4))
        fig.suptitle(f'QQ Plot - {metrica}', fontsize=16)
        
        for i, tipo in enumerate(data_type):
            resultado = data_frame[data_frame['data_type'] == tipo]
            datos = resultado[metrica]
            ax = pg.qqplot(datos, dist='norm', ax=axs[i])
            ax.set_title(f'Tipo: {tipo}')
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(f'qqplot_{metrica}.png')
        plt.close()

metricas = ['accuracy','micro_avg_precision','micro_avg_recall','micro_avg_f1','macro_avg_precision','macro_avg_recall','macro_avg_f1']
csv = '/home/jesus/SugarCane/Pruebas_Estadisticas/DenseNet121/Graficas_DenseNet121/Data/Metrics_SP_DenseNet121.csv'
data_frame = pd.read_csv(csv)
data_type = data_frame['data_type'].unique()
test_normalidad(data_frame, metricas, data_type)