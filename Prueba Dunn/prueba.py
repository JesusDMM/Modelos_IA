import scikit_posthocs as sp
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap

datos = pd.read_csv('/home/jesus/SugarCane/Informacion_general.csv')

datos_test = datos[datos['data_type'] == 'test'].copy()

datos_test['Modelo'] = datos_test['Modelo'].replace('EfficientNetV2B0 Pre-trained', '1')
datos_test['Modelo'] = datos_test['Modelo'].replace('DenseNet121 Pre-trained', '2')
datos_test['Modelo'] = datos_test['Modelo'].replace('ResNet101V2 Pre-trained', '3')

datos_test['Modelo'] = datos_test['Modelo'].replace('EfficientNetV2B0 Non-pre-trained', '4')
datos_test['Modelo'] = datos_test['Modelo'].replace('DenseNet121 Non-pre-trained', '5')
datos_test['Modelo'] = datos_test['Modelo'].replace('ResNet101V2 Non-pre-trained', '6')

p_values = sp.posthoc_dunn(datos_test, val_col='accuracy', group_col='Modelo', p_adjust='holm')

fig, ax = plt.subplots(figsize=(12, 10))

colors = ['firebrick', 'mediumblue']
cmap = ListedColormap(colors)

mask_red = p_values < 0.05
mask_blue = p_values >= 0.05

color_matrix = np.where(mask_red, 0, 1)

mask_triu = np.triu(np.ones_like(p_values, dtype=bool), 1)

sns.heatmap(np.round(p_values, 3), mask=mask_triu, annot=True, cmap=cmap, center=.05, 
            square=True, linewidths=.5, cbar=False, ax=ax, 
            annot_kws={"size": 22}, fmt=".3f", 
            )

ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

red_patch = mpatches.Patch(color='firebrick', label='pval < .05')
blue_patch = mpatches.Patch(color='mediumblue', label='pval ≥ .05') 
plt.legend(handles=[red_patch, blue_patch], loc='upper right', bbox_to_anchor=(0.95, 0.85), fontsize=14)

ax.set_title("Dunn posthoc test's p-values", fontsize=18)

plt.tight_layout()
plt.savefig('Dunn-Bongerroni.png', bbox_inches='tight')
