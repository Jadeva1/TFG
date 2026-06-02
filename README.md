Buenas,

En este repositorio se encuentran los principales scripts que he usado para mi TFG: Espectroscopía de vientos galácticos en galaxias cercanas.

Ha sido un trabajo de muchos meses y esta es una seleción de los códigos y archivos que se pueden calificar como finales, y que permiten reproducir los resultados que he obtenido en mi trabajo.

El código en principio no se pensó para ser lanzado en un ordenador que no fuera el mio así que hay variables al comienzo de algunas funciones con directorios que están puestos directamente con el path que a mi me convenía
en el momento y que habrá que cambiar si alguien quiere reproducirlos en su ordenador.

--------Breve descripción de que hace cada archivo------

- archivos.fits -> Son los resultados del ajuste con pPXF, se utilizan en el resto de archivos como información
- ajuste_nube.py -> Es el script encargado del complejo H alfa y [N II], absolutamente todo lo que se obtiene a partir de estas líneas en mi trabajo, mapas de velocidades,
dispersiones, flujos, SNR, nacen aquí. También carga toda esta información en ficheros .fits para poder usarla en otros sripts.
- Hb_corregido.py -> Lo mismo que ajuste_nube pero para el complejo H beta y [O III]

- - En la carpeta scripts secundarios se encuentran los siguientes archivos responsables de representar distintos plots del trabajo.
- mapas.py -> Extrae la información de los archivos .fits resultantes de hacer el ajuste con pPXF y plotea los mapas para poder compararlos con los de MEGADES
- Resul.py -> Da la tabla de resultados que presento en el TFG para las propiedades físicas del outflow ionizado, utilizando los .fits con los mapas de archivos anteriores.
- comparaciones.py -> Plotea los datos que hemos obtenido para nuestra galaxia frente a los de otras muchas galaxias de la literatura
- BPT.py -> plotea 2 diagramas BPT, uno para la componente ancha y otro para la estrecha, necesita los mapas.fits de scripts anteriores.


Si alguien quisiera replicar mis resultados valdria con ejecutar estos scripts en el orden que aparecen asegurandose de editar en ellos los directorios donde 
tengan los .fits que quieran analizar.

¡Aviso importante!
Los scripts ajuste_nube.py y Hb_corregido.py generan muchas imágenes de spaxels para diagnostico, no ocupan mucho pero si no quereis que se os
llene el directorio de .png de spaxels os recomiendo no ejecutar la funcion plot_spaxels_with_residuals dentro de estos archivos.
