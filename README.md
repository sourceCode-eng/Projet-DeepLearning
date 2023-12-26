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

![Example Image](images/class%20names.png)

Elle contient un total de 9000 images réparties en ensembles d’entraînement et de test.
Chaqu’une composé en deux classes
![Example Image](images/class%20counts.png)

Les données de test sont composés comme suit :

![Example Image](images/class%20proportions%20in%20test%20data.png)

Les données d’entrainement sont présentés comme suit :

![Example Image](images/class%20proportions%20in%20train%20data.png)

## Visualiser quelques données

### Visualisation des données d’entrainement

![Example Image](images/8%20examples%20from%20train%20data.png)

### Visualisation des données de test

![Example Image](images/8%20examples%20from%20test%20data.png)

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
convergence, évite le surajjustement, et offre de meilleures performances en termes de précision et de
matrices de confusion.

### Transfer learning avec VGG (unfrozen base) avec et sans data augmentation

** code **
```
# Charger le modèle VGG16 avec une base dégelée
vgg_base_unfrozen = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
# Créer un modèle séquentiel avec la base VGG16 dégelée
transfer_vgg_unfrozen = Sequential([
 vgg_base_unfrozen,
 Flatten(),
 Dense(256, activation='relu'),
 Dense(128, activation='relu'),
 Dense(2, activation='softmax')
])
# Dégel des dernières couches de la base VGG16 pour fine-tuning
for layer in transfer_vgg_unfrozen.layers[0].layers[-5:]:
 layer.trainable = True
# Compiler le modèle
transfer_vgg_unfrozen.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Entraîner le modèle
history_transfer_vgg_unfrozen = transfer_vgg_unfrozen.fit(train_data, epochs=epochs,
validation_data=test_data)
```

**Description du code **
Chargement de la base VGG16 dégelée : La première étape consiste à charger la base VGG16 préentrainée à partir de l'ensemble de données ImageNet tout en excluant la couche fully connected
(include_top=False) et en spécifiant la forme d'entrée.
Création du modèle séquentiel : On crée un modèle séquentiel auquel on ajoute la base VGG16 dégelée,
suivi d'une couche Flatten pour aplatir les features et des couches fully connected personnalisées pour
la classification.
Dégel des dernières couches : On dégèle les dernières couches de la base VGG16 en itérant sur les
dernières couches du modèle et en les marquant comme entraînables. Cela permet d'ajuster les poids
de ces couches pendant l'entraînement.
Compilation du modèle : On compile le modèle en spécifiant l'optimiseur, la fonction de perte et les
métriques.
Entraînement du modèle : On entraîne le modèle sur les données d'entraînement avec le jeu de
validation.
**Comparaison**
Code : Les deux modèles utilisent la même architecture de base VGG16 avec des couches
supplémentaires adaptées à la tâche spécifique.
Architecture : La seule différence réside dans l'utilisation ou non de l'augmentation de données.
L'augmentation de données permet généralement de régulariser le modèle et d'améliorer la
généralisation en introduisant une variabilité supplémentaire dans les données d'entraînement.
Résultats : Les résultats d'évaluation montrent que le modèle avec augmentation de données a
tendance à mieux généraliser, avec des valeurs de perte et d'accuracy plus stables sur les données de
test. Cependant, le modèle sans augmentation de données atteint également de bonnes performances.
Conclusion : L'utilisation de l'augmentation de données semble bénéfique dans ce scénario, car elle aide
le modèle à mieux généraliser aux données de test. Cela peut être particulièrement utile lorsque la taille
du jeu de données est limitée.

### Transfer Learning using ResNet50 avec et sans data augmentation

#### Code
```
#Chargement du modèle pré-entraîné
#Utilisation de ResNet50 avec weights='imagenet' pour charger les poids pré-entraînés sur ImageNet.
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width,3))
#Construction du modèle de transfert d'apprentissage :
#Ajout de nouvelles couches au-dessus du modèle de base.
#Gel des couches du modèle de base pour éviter la mise à jour des poids pré-entraînés lors del'entraînement initial.
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))
```
#### Comparaison
Les deux modèles montrent de bonnes performances en termes de diminution de la perte et
d'augmentation de la précision.
Le modèle avec augmentation de données a une précision de test plus élevée (0.94) par rapport au
modèle sans augmentation (0.93).
Les deux modèles semblent bien généraliser, mais le modèle avec augmentation de données a une
meilleure capacité à reconnaître des motifs plus variés et complexes dans les données de test.
#### Choix
Dans la plupart des cas, le modèle avec augmentation de données est préférable car il offre
généralement une meilleure généralisation et une résilience accrue face à la variabilité des données.

### Quel est le meilleur modele pour transfert learning

Le choix du meilleur modèle pour le transfert d'apprentissage dépend des caractéristiques spécifiques
de la tâche et des données disponibles. Cependant, voici résumé sur les modèles étudiés :
**VGG (Frozen Base) avec Augmentation de Données :**
Points forts : Bonne précision d'entraînement et de test, augmentation de données pour améliorer la
généralisation.
Points faibles : La courbe de test loss montre une certaine oscillation, mais cela peut être acceptable en
fonction du seuil de tolérance pour la performance.
**VGG (Frozen Base) sans Augmentation de Données :**
Points forts : Haute précision d'entraînement.
Points faibles : La courbe de test loss oscille, ce qui peut indiquer une certaine instabilité. La
performance de test atteint une précision décente.
**VGG (Unfrozen Base) avec Augmentation de Données :**
Points forts : Bonne précision d'entraînement et de test, augmentation de données pour améliorer la
généralisation.
Points faibles : La courbe de test loss montre une certaine oscillation, mais la performance est
généralement bonne.
**VGG (Unfrozen Base) sans Augmentation de Données :**
Points forts : Haute précision d'entraînement.
Points faibles : La courbe de test loss oscille, mais la performance de test atteint une précision décente.
**ResNet (Frozen Base) avec Augmentation de Données :**
Points forts : Bonne précision d'entraînement et de test, augmentation de données pour améliorer la
généralisation.
Points faibles : La courbe de test loss montre une certaine oscillation, mais la performance est
globalement bonne.
**ResNet (Frozen Base) sans Augmentation de Données :**
Points forts : Haute précision d'entraînement.
Points faibles : La courbe de test loss oscille, mais la performance de test atteint une précision décente

## Comparaison des modeles
### Pour le cas sans augmentation de donnees

![Example Image](images/Training%20accuracy%20improvement%20through%20epochs%20without%20data%20augmentation.png)

L'examen de la courbe d'amélioration de l'accuracy d'entraînement à travers les epochs offre des
insights significatifs sur les performances des différents modèles. Le modèle CNN simple se distingue
avec une accuracy remarquablement élevée de 0.98, suggérant une capacité exceptionnelle à apprendre
et à s'adapter aux caractéristiques des données d'entraînement. Cette performance élevée peut être
attribuée à la capacité intrinsèque des CNN à extraire des motifs spatiaux et hiérarchiques dans les
données, ce qui se traduit par une représentation plus riche et discriminante des caractéristiques.
D'un autre côté, le modèle ANN complexe affiche une accuracy d'entraînement relativement plus faible,
plafonnant à 0.9. Cette observation pourrait indiquer que l'architecture plus complexe du réseau de
neurones n'a pas nécessairement conduit à une amélioration significative de la capacité d'apprentissage
sur les données d'entraînement. Il est possible que la complexité accrue du modèle ne se traduise pas
toujours par des performances supérieures, et cela peut également être dû à des contraintes de
données ou d'autres facteurs spécifiques au problème.

![Example Image](images/Validation%20accuracy%20improvement%20through%20epochs%20without%20data%20augmentation.png)

L'analyse des courbes d'amélioration de la précision de transfert sur la validation, en particulier dans le
contexte de modèles VGG simple et ANN complexe sans augmentation de données, révèle des
tendances significatives.
Le modèle simple VGG affiche une performance exceptionnelle avec une précision de transfert sur la
validation atteignant 0.98. Cette excellente précision peut être attribuée à l'efficacité de l'architecture
VGG dans l'extraction de caractéristiques complexes des données, même sans l'ajout de données
augmentées. La simplicité de l'architecture VGG, combinée à la puissance des couches convolutives,
semble être un atout dans ce contexte, permettant au modèle de généraliser efficacement sur les
données de validation.
D'autre part, le modèle ANN complexe présente une précision de transfert sur la validation légèrement
inférieure, évaluée à 0.88. Cette différence peut s'expliquer par la complexité accrue du modèle ANN et
sa sensibilité potentielle au surajustement. La précision légèrement inférieure peut être due à la
difficulté du modèle à généraliser de manière optimale sur les données de validation, en particulier sans
l'utilisation de données augmentées.

![Example Image](images/epoch%20times%20for%20each%20model%20without%20data%20augmentation.png)

Le tranffert_vgg model , tranffert_vgg model_unfrozen et transfert_resnet_model présentent les
d’époches les plus élevéé puisqu’ils possèdent le nombre de paramètres les plus élevés comme le
montre le diagramme suivant

![Example Image](images/number%20of%20parameters%20in%20each%20model%20without%20data%20augmentation.png)

### Pour le cas avec augmentation de donnees

![Example Image](images/Training%20accuracy%20improvement%20through%20epochs%20with%20data%20augmentation.png)

Les courbes d'apprentissage sont des outils essentiels pour évaluer les performances des modèles de
machine learning au fil des epochs. Dans le cadre de cette étude comparative entre le modèle de
transfert ResNet et le modèle ANN simple, plusieurs observations clés peuvent être extraites de la
courbe représentant l'amélioration de la précision d'entraînement au fil des epochs.
Le modèle de transfert ResNet affiche une tendance positive marquée, atteignant une précision
d'entraînement remarquable de 0.95. Cette performance élevée peut être attribuée à la puissance de la
méthode de transfert learning, qui permet au modèle de bénéficier des connaissances préalables
apprises sur des données volumineuses. La capacité du ResNet à extraire des caractéristiques complexes
à partir des données semble se traduire par une précision accrue au fil de l'entraînement.
En revanche, le modèle ANN simple présente une performance moins élevée avec une précision
d'entraînement de 0.83. Cela pourrait indiquer des limitations dans la capacité du modèle à apprendre
des représentations complexes et des motifs abstraits dans les données, en particulier par rapport à la
méthode plus sophistiquée de transfert learning utilisée par le ResNet.

![Example Image](images/Validation%20accuracy%20improvement%20through%20epochs%20with%20data%20augmentation.png)


Le modèle simple VGG affiche une performance exceptionnelle, atteignant une précision de validation
maximale de 0.98. Cette observation confirme la robustesse de l'architecture VGG, même dans des
conditions d'augmentation de données. La capacité du modèle à maintenir une précision élevée
témoigne de son aptitude à généraliser efficacement sur les données de validation, même lorsque
celles-ci sont augmentées.



En revanche, le modèle ANN complexe présente une précision de validation maximale légèrement
inférieure, évaluée à 0.88. Cette disparité peut être attribuée à la complexité accrue du modèle ANN,
qui pourrait être plus sensible au surajustement, malgré l'apport bénéfique des données augmentées. La
nature complexe de l'architecture peut rendre le modèle plus susceptible de varier dans ses
performances au fil des époques.

![Example Image](images/epoch%20times%20for%20each%20model%20with%20data%20augmentation.png)


Le tranffert_vgg model , tranffert_vgg model_unfrozen et transfert_resnet_model présentent les
d’époches les plus élevéé puisqu’ils possèdent le nombre de paramètres les plus élevés comme le
montre le diagramme suivant

![Example Image](images/number%20of%20parameters%20in%20each%20model%20with%20data%20augmentation.png)

