# Антиплагиат
Задание 6 для отбора на курс Тинькофф Поколение

В качестве решения использую эмбеддинги из word2vec и решаю задачу используя triplet loss(используя косинусное расстояние)

### word2vec.model - обученный word2vec, чтобы получить эмбеддинги текста
### model.pkl - модель,обученная на triplet loss
### train.py - обучение модели,параметры запуска:
##### files,files2,files3 - файлы должны лежать в папке data
##### --model model.pkl - путь к сохраняемой модели
### compare.py - инференс модели,параметры запуска:
##### input.txt - файл,в котором в каждой строчке через пробел лежат файлы для сравнения
##### scores.txt - файл в котором будут лежать результаты сравнения
##### --model model.pkl - путь к используемой модели

