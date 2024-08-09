import os
import shutil

carpeta_origen = '/home/jesus/SugarCane/Pruebas_Estadisticas/EfficientNetV2B0/Graficas_EfficientNetV2B0/Metricas/'
carpeta_destino = '/home/jesus/SugarCane/Pruebas_Estadisticas/EfficientNetV2B0/Graficas_EfficientNetV2B0/Metricas/Densidad sin pesos/'

inicio_patron = 'Metrica_'
fin_patron = 'sp.png'

for archivo in os.listdir(carpeta_origen):
    if archivo.startswith(inicio_patron) and archivo.endswith(fin_patron):
        ruta_archivo = os.path.join(carpeta_origen, archivo)
        shutil.move(ruta_archivo, carpeta_destino)

print("Archivos movidos exitosamente.")