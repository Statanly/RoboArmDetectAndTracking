# Управление роботизированной рукой с помощью компьютерного зрения

## Подготовка камер
1. Скачать приложение [Eye4](https://www.vstarcam.com/software-download).
2. Нажать “добавить камеру”, отсканировать QR-код. Будет выполняться звуковое подключение.
Если не подключается - сбросить камеру до заводских настроек через нажатие на кнопку, утопленную сзади (скрепкой/зубочисткой) для сброса. 
После этого она обычно без проблем подключается. 
3. Установить в настройках камер пароль для подключения.
4. Получить  IP камер через любой сканер IP
Составить такие адреса: ```rtsp://admin:{пароль, который задается через приложение}@{IP камеры}:10554/tcp/av0_0```

NB! Камеры и компьютер должны находиться в одной сети Wi-Fi.

## Подготовка окружения
1. Скачать репозиторий с помощью ```git clone```.
2. Перейти в корневую директорию RoboArmDetectingAndTracking (```cd RoboArmDetectingAndTracking```).
   1. Скачать необходимые библиотеки для управления рукой:
      Для прямого управления выполнить следующие команды в терминале (не протестировано):
      ```
      git clone https://github.com/Promobot-education/Rooky.git
      cd ./Rooky
      chmod +x ./install.sh
      ./install.sh
      ```
      Для управление с помощью ROS:
      ```
       sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
       sudo apt install curl # Если curl не установлен
       curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
       sudo apt-get update
       sudo apt-get install ros-kinetic-desktop-full
       sudo apt install python-rosdep
       sudo rosdep init
       rosdep update
       echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
       source ~/.bashrc
       ```
      Установить пакет с серверным ПО promobot-edu-control. Добавить в систему ppa репозиторий (если не добавлен)
       ```
       curl -s --compressed "https://Promobot-education.github.io/ppa/KEY.gpg" | sudo apt-key add -
       sudo curl -s -o /etc/apt/sources.list.d/promobot-education.list "https://Promobot-education.github.io/ppa/promobot-education.list"
       ```
       Установить пакет:
       ```
       sudo apt update
       sudo apt install promobot-edu-control
       ```
       Добавить путь до файлов серверного ПО
       ```
       echo "source /opt/promobot/EduControl/install/setup.bash" >> ~/.bashrc
       source ~/.bashrc
       ```
       Проверить зависимости
       ```
       cd /opt/promobot/EduControl
       rosdep install --from-paths install --ignore-src -r -y
       ```
      Приведенные команды протестированы для Ubuntu 16.04. Для установки на Windows смотреть:
   https://promobot-education.github.io/Rooky/
3. Установить необходимые зависимости с помощью

    `pip install -r requirements.txt`
4. Скачать [веса](https://drive.google.com/file/d/1kRdr_eiMOOf0Nd5_1dUTuhqwnNPYDaoU/view?usp=sharing) и распаковать их в корневую папку.

## Запуск

`python main.py --source-front {адрес  передней камеры} --source-side {адрес боковой камеры} `

Основные параметры:

`--show-vid` - показывать видео с детекцией в режиме реального времени

`--ros` - использовать ROS

`--not-move-arm` - провести только детекцию и расчёт расстояния без управления рукой
