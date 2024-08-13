import numpy as np
import pingouin as pg
import pandas as pd 

def test_normalidad(data_frame, metricas, data_type):
    all_results = pd.DataFrame()
    
    for tipo in data_type:
        resultado = data_frame[data_frame['data_type'] == tipo]
        for metrica in metricas:
            test = pg.normality(resultado[metrica], method = 'shapiro')
            test.insert(0, 'Metrica', metrica)
            test.insert(1, 'Data_Type', tipo)
            all_results = pd.concat([all_results, test], ignore_index=True)
    all_results.to_csv('Prueba_normal_efficientnet_sp_shapirowilk.csv', index=False)

metricas = ['accuracy','micro_avg_precision','micro_avg_recall','micro_avg_f1','macro_avg_precision','macro_avg_recall','macro_avg_f1']
csv = '/home/jesus/SugarCane/Pruebas_Estadisticas/EfficientNetV2B0/Graficas_EfficientNetV2B0/Data/Metricas_Efficientnet_sp.csv'
data_frame = pd.read_csv(csv)
data_type = data_frame['data_type'].unique()
test_normalidad(data_frame, metricas, data_type)