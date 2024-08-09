import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import os.path
import pandas as pd

def secciones(folder_path):
    carpetas_con_conteo = {}

    for root, dirs, archivo in os.walk(folder_path):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            file_count = len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])
            carpetas_con_conteo[dir_name] = file_count
    
    return carpetas_con_conteo


def plot_image_classes(df, category):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Categoria', y='Cantidad' , palette='viridis')
    plt.title(f'{category}')
    plt.xlabel('Tipo')
    plt.ylabel('Numero de imagenes')
    plt.xticks(rotation=70)
    plt.savefig("Grafica.png", bbox_inches='tight')
    plt.show()


train_dir = '/home/Enfermedad/'
carpetas_con_conteo = secciones(train_dir)
print(carpetas_con_conteo)

df = pd.DataFrame(list(carpetas_con_conteo.items()), columns=['Categoria', 'Cantidad'])

plot_image_classes(df, 'Visualización de cantidad de imagenes por clase')