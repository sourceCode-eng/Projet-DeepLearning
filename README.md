# COMPTE RENDU PROJET DEEP LEARNING : FACE MASK CLASSIFICATION

**Présenté par:**
- Mahdi Ben Slima
- Abid Oussema

**Année Scolaire:**
2023-2024

## Contents
1. [Présentation de la base](#presentation-de-la-base)
2. [Visualiser quelques données](#visualiser-quelques-donnees)
   - 2.1 [Visualisation des données d’entrainement](#visualisation-des-donnees-dentrainement)
   - 2.2 [Visualisation des données de test](#visualisation-des-donnees-de-test)
3. [Modèle from scratch](#modele-from-scratch)
   - 3.1 [Comparaison des ANN](#comparaison-des-ann)
     - 3.1.1 [Comparaison entre le Modèle ANN Simple sans et avec data augmentation](#comparaison-entre-le-modele-ann-simple-sans-et-avec-data-augmentation)
     - 3.1.2 [Comparaison ANN complexe avec et sans data augmentation](#comparaison-ann-complexe-avec-et-sans-data-augmentation)
     - 3.1.3 [Quel est le meilleur modèle entre les ANN](#quel-est-le-meilleur-modele-entre-les-ann)
   - 3.2 [Comparaison des CNN](#comparaison-des-cnn)
     - 3.2.1 [Comparaison entre CNN simple sans et avec data augmentation](#comparaison-entre-cnn-simple-sans-et-avec-data-augmentation)
     - 3.2.2 [Comparaison des CNN complexe sans et avec data augmentation](#comparaison-des-cnn-complexe-sans-et-avec-data-augmentation)
     - 3.2.3 [Quel est le meilleur modèle entre les CNN](#quel-est-le-meilleur-modele-entre-les-cnn)
4. [Transfer learning](#transfer-learning)
   - 4.1 [Transfer learning avec VGG (frozen base) avec et sans data augmentation](#transfer-learning-avec-vgg-frozen-base-avec-et-sans-data-augmentation)
     - 4.1.1 [Les points importantes dans le code](#les-points-importantes-dans-le-code)
     - 4.1.2 [Comparaison](#comparaison)
   - 4.2 [Transfer learning avec VGG (unfrozen base) avec et sans data augmentation](#transfer-learning-avec-vgg-unfrozen-base-avec-et-sans-data-augmentation)
     - 4.2.1 [Code](#code)
     - 4.2.2 [Description du code](#description-du-code)
     - 4.2.3 [Comparaison](#comparaison)
   - 4.3 [Transfer Learning using ResNet50 avec et sans data augmentation](#transfer-learning-using-resnet50-avec-et-sans-data-augmentation)
     - 4.3.1 [Code](#code)
     - 4.3.2 [Comparaison](#comparaison)
     - 4.3.3 [Choix](#choix)
   - 4.4 [Quel est le meilleur modèle pour transfert learning](#quel-est-le-meilleur-modele-pour-transfert-learning)
5. [Comparaison des modèles](#comparaison-des-modeles)
   - 5.1 [Pour le cas sans augmentation de données](#pour-le-cas-sans-augmentation-de-donnees)
   - 5.2 [Pour le cas avec augmentation de données](#pour-le-cas-avec-augmentation-de-donnees)

## Présentation de la base

Cette base de données est divisée sur deux classes, « with mask», « without mask». 
teswira
Elle contient un total de 9000 images réparties en ensembles d’entraînement et de test.
Chaqu’une composé en deux classes
teswira
Les données de test sont composés comme suit :
teswira
Les données d’entrainement sont présentés comme suit :
teswira

## Visualiser quelques données

### Visualisation des données d’entrainement
teswira

### Visualisation des données de test
teswira

## Modèle from scratch

### Comparaison des ANN

#### Comparaison entre le Modèle ANN Simple sans et avec data augmentation

**Architecture :**
Les deux modèles ont la même architecture, mais le modèle avec augmentation de données peut traiter
une variété plus importante d'images grâce à la diversité introduite par l'augmentation.
**Courbes d'Apprentissage :**
Le modèle avec augmentation de données montre une diminution plus rapide de la perte d'entraînement
et une légère amélioration de la perte de test par rapport au modèle sans augmentation.
**Courbes d'Accuracy :**
Le modèle avec augmentation de données montre une amélioration de l'accuracy sur l'ensemble de test
par rapport au modèle sans augmentation.
**Matrice de Confusion :**
Le modèle avec augmentation de données montre une amélioration dans la détection des vrais positifs
et une diminution des faux négatifs par rapport au modèle sans augmentation.
**Conclusion :**
L'ajout de l'augmentation de données semble bénéfique, car le modèle avec augmentation de données
présente de meilleures performances en termes de courbes d'apprentissage, d'accuracy, et de matrice de
confusion par rapport au modèle sans augmentation.

#### Comparaison ANN complexe avec et sans data augmentation

**Architecture :**
Structure complexe : L'ANN complexe a une structure plus profonde et complexe avec davantage de
couches par rapport ANN simple.
Plus de couches cachées : Il peut y avoir davantage de couches cachées dans le modèle. Dans l'exemple
ci-dessus, il y a plusieurs couches cachées, y compris des couches de suppression (Dropout) pour réduire
le surajustement.
Davantage de fonctions d'activation : Les couches cachées peuvent utiliser différentes fonctions
d'activation, telles que relu, sigmoid, etc.
Utilisation de Dropout : La couche de suppression (Dropout) est utilisée pour régulariser le modèle et
éviter le surajustement en désactivant aléatoirement certains neurones pendant l'entraînement.
**Courbes d'Apprentissage :**
Les deux modèles présentent des courbes de perte similaires, avec des légères variations. La
convergence est atteinte dans des plages comparables.
Courbes d'Accuracy :
Le modèle avec augmentation de données atteint une précision de test plus élevée, même si la précision
de l'ensemble d'entraînement est déjà élevée. Cela suggère une meilleure généralisation du modèle
augmenté.
**Matrice de Confusion :**
Les deux modèles ont des performances assez similaires en termes de matrice de confusion. Cependant,
le modèle avec augmentation de données a une légère amélioration du nombre de vrais positifs.
**Conclusion**
L'ajout d'augmentation de données semble bénéfique pour le modèle ANN complexe dans ce cas
particulier. Bien que les améliorations ne soient pas radicales, le modèle avec augmentation de données
montre une tendance à mieux généraliser, ce qui se traduit par une précision de test plus élevée.
L'augmentation de données peut aider à rendre le modèle plus robuste et moins

#### Quel est le meilleur modèle entre les ANN

**Analyse Comparative :**
Le modèle ANN Simple avec augmentation de données semble avoir une meilleure précision de test
(0.83) par rapport au modèle sans augmentation de données (0.92). Cependant, le modèle sans
augmentation de données a une précision de test déjà assez élevée (0.92).
Pour le modèle ANN Complexe, la comparaison est plus nuancée. Le modèle avec augmentation de
données a une meilleure précision de test (0.82) par rapport au modèle sans augmentation de données
(0.88). Cependant, le modèle sans augmentation de données a déjà une précision de test assez élevée
(0.88).
**Conclusion :**
Si l'objectif principal est la précision de test, le modèle ANN Simple avec augmentation de données
semble être le meilleur parmi ceux que vous avez évalués.

### Comparaison des CNN
#### Comparaison entre CNN simple sans et avec data augmentation

**Code et Architecture :**
Les deux modèles utilisent une architecture de CNN simple avec une ou deux couches de convolution,
des couches de pooling, et des couches entièrement connectées à la fin.
Le modèle avec data augmentation utilise probablement une couche ImageDataGenerator pour
augmenter la diversité des données pendant l'entraînement.
**Résultats :**
Les deux modèles semblent bien apprendre les données, montrant une diminution de la perte et une
augmentation de l'exactitude.
Le modèle avec data augmentation semble mieux généraliser aux données de test, comme indiqué par
une meilleure performance sur la courbe de test accuracy et une matrice de confusion avec des valeurs
plus élevées.
**Meilleur Modèle :**
Le modèle avec data augmentation semble être meilleur, car il a une meilleure performance sur les
données de test, ce qui indique une meilleure capacité à généraliser à de nouvelles données.
En conclusion, l'utilisation de la data augmentation semble bénéfique pour améliorer la performance du
modèle sur des données non vues, ce qui en fait une approche recommandée pour améliorer la
généralisation des modèles CNN.

#### Comparaison des CNN complexe sans et avec data augmentation

**Code et Architecture :**
Les deux modèles utilisent une architecture de CNN complexe avec plusieurs couches de convolution,
des couches de pooling, et des couches entièrement connectées à la fin.
Le modèle avec data augmentation utilise probablement une couche ImageDataGenerator pour
augmenter la diversité des données pendant l'entraînement.
**Résultats :**
Les deux modèles montrent une amélioration de la performance par rapport au temps, avec une
diminution de la perte et une augmentation de l'exactitude.
**Meilleur Modèle :**
Le modèle avec data augmentation semble être meilleur, car il atteint une accuracy plus élevée sur les
données de test et présente une diminution plus significative de la perte.
En conclusion, l'utilisation de la data augmentation semble également bénéfique pour le modèle CNN
complexe, montrant une amélioration dans la capacité à généraliser aux données de test.

#### Quel est le meilleur modèle entre les CNN

Le CNN Complexe avec Data Augmentation semble avoir les meilleures performances, avec la précision
la plus élevée sur les données de test (~93%) et une courbe de test loss en diminution.
Le CNN Simple avec Data Augmentation montre également de bonnes performances, mais légèrement
inférieures à celles du CNN Complexe avec Data Augmentation.
Les CNN sans data augmentation montrent des performances légèrement inférieures en termes de
précision.
**Conclusion :**
Le CNN Complexe avec Data Augmentation semble être le meilleur modèle parmi ceux évalués ici en
raison de ses performances supérieures sur les données de test. Cependant, il est important de noter
que le choix du meilleur modèle dépend souvent des caractéristiques spécifiques de votre problème et
de la nature de vos données


## Transfer learning
### Transfer learning avec VGG (frozen base) avec et sans data augmentation

**Les points importantes dans le code**
Utilisation de la base VGG16 pré-entraînée sur ImageNet en tant que couche de base, gelée pour ne pas
être réentraînée.
```
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
Geler la base VGG16: La base VGG16 est gelée pour ne pas mettre à jour ses poids pendant
l'entraînement.
base_model.trainable = False
```
**Comparaison**
Courbes de Loss (Entraînement et Test) : Le modèle avec augmentation de données montre une
meilleure convergence de la loss pendant l'entraînement et une meilleure généralisation sur le jeu de
test. Il évite également l'oscillation observée dans le modèle sans augmentation.
Courbes d'Accuracy (Entraînement et Test) : Le modèle avec augmentation de données atteint une
précision plus élevée sur le jeu de test et évite le surajustement observé dans le modèle sans
augmentation.
Matrices de Confusion : Les résultats de True Positive, True Negative, False Positive et False Negative
indiquent que le modèle avec augmentation de données a de meilleures performances globales.
Conclusion :
Le modèle avec augmentation de données semble être meilleur, car il atteint une meilleure
convergence, évite le suraj

### Transfer learning avec VGG (unfrozen base) avec et sans data augmentation
Your content here...

### Transfer Learning using ResNet50 avec et sans data augmentation
Your content here...

### Quel est le meilleur modèle pour transfert learning
Your content here...

## Comparaison des modèles
### Pour le cas sans augmentation de données
Your content here...

### Pour le cas avec augmentation de données
Your content here...
