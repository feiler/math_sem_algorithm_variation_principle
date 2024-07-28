# math_sem_algorithm_variation_principle
For the MathSem, we explored the variation principle. I chose a theme related to informatics. In this repository, I am sharing my scripts so that others can use them to test the examples described in the book.


# Deutsch
# Anleitung
Die Installationsanleitung ist für Windows mit VS-Code und Docker aufgebaut (könnte auch mit Apple funktionieren, einfach ohne WSL). 
Die Idee hinter der Installation ist, dass auch Personen, welche keine Erfahrungen mit Entwicklungen haben.
Die Software starten können.
1. [Download Projekt von Github](#projekt-ohne-git-herunterladen)
2. [Install WSL2](#installation-wsl2)
3. [Install VS-Code](#installation-docker)
4. [Install VS-Code](#installation-vs-code)
5. [Öffne das Projekt](#öffne-das-projekt)

## Projekt Ohne Git herunterladen
![Download From Git](/images/download_from_git.png)

## Installation WSL2
WSL2 bringt Linux auf Windows, dies wird für Docker gebraucht. Die Anleitung kann unter diesem Link gefunden werden https://learn.microsoft.com/en-us/windows/wsl/install.

Anleitung in Kurz:

Drücke start und suche nach Powershell, öffne diese als Administrator. 
![Open PowerShell](/images/open_powershell.png)
Gib folgenden Befehle ein.
```
wsl --install -d Ubuntu-22.04
```

Danach öffne die Ubuntu Distributation. Bei der Installation wird nach einem Benutzername und ein Password eingeben. Achtung beim Password wird sieht man nicht das was eingegeben wurde, aber die Buchstaben werden eingeben.
![Open PowerShell](/images/ubuntu.png)

## Installation Docker
Mit Docker lassen sich Entwicklungsumgebungen einfacher Teilen, was es für alle einfacher macht. Docker kann unter diesem Link heruntergeladen https://www.docker.com/

Nach Installation starte Docker und überprüfe ob der Hacken bei WSL gesetzt.
![Docker Config](/images/docker_check_config.png)

## Installation VS-Code
VS-Code ist eine freie IDE (Integrated Development Environment), dies kann unter diesem Link heruntergeladen werden.

https://code.visualstudio.com/

Nach Installation gehe auf Extensions und suche nach Dev Container.
![Docker Config](/images/vs_code_install_extension.png)

## Öffne das Projekt
Öffne den heruntergeladenen Ordner.
![Open Folder](/images/open_project.png)

Wichtig Docker muss laufen, danach hat es in der unteren Ecke ein blaues Feld.
![Open Folder](/images/open_project_dev_container.png)

## Start Scripts


Dies sollt
# TODO
- Anleitung VS-Code install
- Docker insall
- WSL2 insall
- Add folder with own Readme
- Anleitung wie gestartet diese werden
- Aufbereitung der Skripts
