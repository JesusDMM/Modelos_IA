import Augmentor

#Instancia donde se van a aumentar las imagenes
direccion = "/home/Enfermedad/Sana/"
p = Augmentor.Pipeline(direccion)

#Acciones que le se van a implementar a las imagenes con su respectiva probabilidad de ocurrir
p.rotate(probability=0.6, max_left_rotation=20, max_right_rotation=20)
p.zoom(probability=0.5, min_factor=1.2, max_factor=1.6)
p.skew(probability=0.5)
p.resize(probability=1.0, width=493, height=493)

#guardar una cantidad de imagenes establecidad
cantidad = 548
p.sample(cantidad)