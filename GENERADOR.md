# Generador de datos

El generador sirve para crear los datos sintéticos con los que entrenamos el modelo.

Cada tarea se guarda en un CSV dentro de `data/` con dos columnas: `prompt`, que es la pregunta, e `y`, que es la respuesta correcta (ambos en formato string).

Ahora mismo se pueden generar seis tareas: sumas y restas, multiplicaciones y divisiones, la paridad de un número, ordenar letras alfabéticamente, deletrear una cadena y sacar la primera y la última letra de una palabra.

En cada función puedes elegir cuántos ejemplos quieres, el rango de los números o la longitud de las cadenas, la semilla y la plantilla del prompt.

En las de mates, `add` decide si salen sumas y restas o multiplicaciones y divisiones.

Con `result=True` la solución también aparece escrita en el prompt, por ejemplo 37 + 12 = 49, y con False solo aparece la pregunta.

Para generarlo todo de golpe basta con ejecutar el notebook, que al final llama a `generate_all`. Para crear muchos dataset simplemente hay que replicar el código ajustandolo para cada caso. El notebook puede ser guarro ya que simplemente va a crear los datos sintéticso para el modelo.

Hay que tener en cuenta que si el CSV ya existe las filas nuevas se añaden al final, así que si quieres empezar de cero tienes que borrarlos antes o cambiar el nombre.

El generador solo crea los CSV, lo demás se hace en otros archivos.

En `dataset.py` se unen todos los CSV en una sola tabla, se quitan los duplicados y se hace el split en train, val y test (80/10/10 por defecto), barajando cada tarea por separado.

En `dataloader.py` se montan los batches y se hace el shuffle del train, siempre con semilla para que todo sea reproducible.


**Importante:** para cada tarea habría que usar una semilla distinta para no sesgar el modelo, para los tres experimentos que vamos a probar deberíamos de usar el mismo dataset para comparar los resultados. Habría que hacer un mini-dataset con todas las tareas en el caso de necesitar un fine-tunning general al final de un experimento.
