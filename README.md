## Описание
#### Папка Modules
В ней хранятся либо переписанные готовые модули, либо модули созданные с нуля.
#### Папка src
В ней хранятся все файлы связанные с проектом (Например: .mp4 .jpg .png).
#### Папка site
~~В ней хранятся все файлы связанные с сайтом.(По крайней мере должны будут)~~
#### Файл .gitignore
Для того, чтобы определить какие файлы и папки не нужно добавлять в git репозиторий.
#### Файл yolov8n.pt
Готовая модель YOLO версии 8n.
#### Файл object_counting_output.avi
Вывод нашей программы, на котором можно увидеть сколько человек зашло, вышло и сколько человек на камере.
#### Файл test_counter_from_file.py
Обработка входного видеопотока с файла **file5.mp4**
#### Файл test_counter_from_webcam.py
Обработка входного видеопотока с камеры.
## Установка зависимостей
Для установки зависимостей нужно прописать следующие команды:
`pip install opencv-python`
`pip install ultralytics`
