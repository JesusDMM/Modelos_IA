import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

def unir_csv (ruta, nombre):
    directorio = ruta
    nombre_final = nombre + '.csv'

    dfs = []

    for archivo in os.listdir(directorio):
        if archivo.startswith("cross_validation_metrics") and archivo.endswith("SP.csv"):
            ruta_archivo = os.path.join(directorio, archivo)
            df = pd.read_csv(ruta_archivo)
            dfs.append(df)

    df_concatenado = pd.concat(dfs, ignore_index=True)

    df_concatenado.to_csv(nombre_final, index=False)

def crear_grafica(csv, nombre, info, nombre_grafico):
    
    datos = pd.read_csv(csv)
    datos = datos[[info[0], info[1], 'epoch']]
    sns.lineplot(data=datos, x='epoch', y=info[0], label=info[0])
    sns.lineplot(data=datos, x='epoch', y=info[1], label=info[1])
    plt.xlabel('Epocas')
    plt.ylabel(info[0]) 
    plt.title(nombre_grafico)
    plt.legend()
    plt.savefig(nombre)
    
def crear_grafica_metricas(csv, nombre, metrica, nombre_grafico):
    
    datos = pd.read_csv(csv)
    datos_entrenamiento = datos[datos['data_type'] == 'train'][[metrica]]
    datos_validacion = datos[datos['data_type'] == 'val'][[metrica]]
    datos_test = datos[datos['data_type'] == 'test'][[metrica]]
    
    sns.kdeplot(data=datos_entrenamiento, x=metrica,label='Entrenamiento')
    sns.kdeplot(data=datos_validacion, x=metrica, label='Validacion')
    sns.kdeplot(data=datos_test, x=metrica, label='Test')
    
    plt.xlabel(metrica)
    plt.ylabel('Tipo')
    plt.title(nombre_grafico)
    plt.legend()
    
    plt.savefig(nombre)
    plt.close()

def crear_grafica_metricas_bloxplot(csv, nombre, metrica, nombre_grafico):
    
    datos = pd.read_csv(csv)
    
    sns.boxplot(data=datos, x=metrica, y='data_type')
    
    plt.xlabel(metrica)
    plt.ylabel('Tipo')
    plt.title(nombre_grafico)
    
    plt.savefig(nombre)
    plt.close()

#ruta = '/home/jesus/SugarCane/Pruebas_Exhaustivas/DenseNet121/'
#nombre = 'Metrics_SP_DenseNet121'
#unir_csv (ruta, nombre)
#unir_csv('../../Pruebas_Exhaustivas/DenseNet121/', 'Metrics_CP_DenseNet121')    
#datos = ['loss', 'val_loss']
#nombre = 'lineplot_loss_sp.png'
#csv = 'Graficas_DenseNet121/Data/History_SP_DenseNet121.csv'
#nombre_grafico = 'Loss'
#crear_grafica(csv, nombre, datos, nombre_grafico)
#def crear_grafica_metricas(csv, nombre, metrica, nombre_grafico):
metricas = ['accuracy', 'micro_avg_precision', 'micro_avg_recall', 'micro_avg_f1', 
            'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1']
csv = 'Graficas_DenseNet121/Data/Metrics_SP_DenseNet121.csv'
for metrica in metricas:
    nombre = 'Metrica_' + metrica + '_sp_Boxplot.png'
    nombre_grafico = metrica
    crear_grafica_metricas_bloxplot(csv, nombre, metrica, nombre_grafico)