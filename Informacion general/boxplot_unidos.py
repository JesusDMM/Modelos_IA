import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def cambiar_nombre(datos):
    modelos = ['EfficientNetV2B0 Pre-trained', 'EfficientNetV2B0',
               'DenseNet121 Pre-trained', 'DenseNet121', 
               'ResNet101V2 Pre-trained', 'ResNet101V2']
                
    df_combinado = pd.DataFrame()
    for id, dato in enumerate(datos):
        df = pd.read_csv(dato)
        df['Modelo'] = modelos[id]
        df_combinado = pd.concat([df_combinado, df], ignore_index=True)
    df_combinado.to_csv('Informacion_general.csv', index=False)

def crear_grafica_metricas_boxplot2(csv):
    datos = pd.read_csv(csv)

    ax = sns.boxplot(data=datos, x="Modelo", y="accuracy", hue="data_type", width=0.6, dodge=True, gap=0.2)
    
    ax.set_xlabel('Model')
    
    plt.xticks(ticks=range(6), labels=[f"{i+1}" for i in range(6)])

    plt.savefig('prueba.png', bbox_inches='tight')
    plt.close()


def crear_grafica_metricas_boxplot(csv):
    datos = pd.read_csv(csv)

    ax = sns.boxplot(data=datos, x="Modelo", y="accuracy", hue="data_type")

    modelos = datos['Modelo'].unique()
    contenido_modelos = [plt.Line2D([0], [0], linestyle='None', label=f"{i+1} {modelo}") for i, modelo in enumerate(modelos)]

    handles_hue, labels_hue = ax.get_legend_handles_labels()
    handles_hue = handles_hue[len(modelos):]
    labels_hue = labels_hue[len(modelos):]

    plt.legend(handles=contenido_modelos, title='Modelos', bbox_to_anchor=(1.05, 0.1), loc='lower left')

    plt.legend(handles=handles_hue, labels=labels_hue, title='Tipo de Datos', bbox_to_anchor=(1.05, 0.1), loc='center left')

    plt.title('Comportamiento de la metrica accuracy para todos las arquitecturas')

    plt.xticks(ticks=range(len(modelos)), labels=[f"{i+1}" for i in range(len(modelos))])

    plt.savefig('prueba2.png', bbox_inches='tight')
    plt.close()
    
datos = [
    '/home/jesus/SugarCane/Pruebas_Estadisticas/EfficientNetV2B0/Graficas_EfficientNetV2B0/Data/Metricas_Efficientnet_cp.csv',
    '/home/jesus/SugarCane/Pruebas_Estadisticas/EfficientNetV2B0/Graficas_EfficientNetV2B0/Data/Metricas_Efficientnet_sp.csv',
    '/home/jesus/SugarCane/Pruebas_Estadisticas/DenseNet121/Graficas_DenseNet121/Data/Metrics_CP_DenseNet121.csv',
    '/home/jesus/SugarCane/Pruebas_Estadisticas/DenseNet121/Graficas_DenseNet121/Data/Metrics_SP_DenseNet121.csv',
    '/home/jesus/SugarCane/Pruebas_Estadisticas/ResNet101V2/Graficas_ResNet101V2/Data/Metrics_CP_ResNet101V2.csv',
    '/home/jesus/SugarCane/Pruebas_Estadisticas/ResNet101V2/Graficas_ResNet101V2/Data/Metrics_SP_ResNet101V2.csv'
]
#cambiar_nombre(datos)
crear_grafica_metricas_boxplot2('Informacion_general.csv')
