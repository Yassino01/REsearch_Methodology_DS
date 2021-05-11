# Static Hand Gestures : 

Etude de **classification** de geste à travers la lecture, la compréhension, et la proposition de modèles et de protocole d’évaluation.


## Datasets :

* [UC2017 Static and Dynamic Hand Gestures](https://zenodo.org/record/1319659#.YJrKRllfgVv)

Il s’agit d’un jeu de données de reconnaissances de gestes à partir de mesures prises par un gant de réalité virtuelle.


Il est constitué de deux sous-datasets :
*  SG - Static Gesture : la position de la main à un instant donné
*  DG - Dynamic Gesture : la série temporelle des positions

Une description plus complète des données, de la procédure d’acquisitions ainsi que des baselines sont disponibles dans l’article [ici](https://www.researchgate.net/publication/330429917_Online_Recognition_o
f_Incomplete_Gesture_Data_to_Interface_Collaborative_Robots)


Pour des raisons de simplicité, on se limitera dans un premier temps au sous-dataset Static Gesture. 

## Analyses des données :

Voici les classes des gestes disponibles dans le dataset :

<p align="center">
  <img src="gestures/figures/hesture_classes.png" width="700" title = "Static gestures" >
</p>

Il y a plusieurs utilisateurs, avec plusieurs mesures par exemple, ce qui pose des problèmes dont il faudra tenir compte. Il n’y a pas de découpage train/test fourni dans le dataset

Nous pouvons classifier les données en 8 classes pour l’utilisateur et en 24 classes pour le type de mouvement. 

Nous allons commencer par analyser les données et notamment leur répartition par rapport à ces class.

<p align="center">
  <img src="gestures/figures/hist_mouv.png" width="600" title = "Répartition des gestes " >
</p>


Et la répartition des gestes par utilisateur.

<p align="center">
  <img src="gestures/figures/nb_exemple_par_user.png" width="600" title = "Répartition des gestes " >
</p>


## Mise en œuvre d’un projet Machine Learning: 
 * chargement 
 * exploration
 * prétraitements
 * formalisation de la tâche,
 * choix des modèles et des méthodes
 * définition d’un protocole de test et d’évaluation de performances.

