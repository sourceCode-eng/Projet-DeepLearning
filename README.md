# COMPTE RENDU PROJET DEEP LEARNING : FACE MASK CLASSIFICATION

**Présenté par:**
- Mahdi Ben Slima
- Abid Oussema

**Année Scolaire:**
2023-2024

## Contents
1. [Présentation de la base](#i-présentation-de-la-base)
2. [Visualiser quelques données](#ii-visualiser-quelques-données)
   - 2.1 [Visualisation des données d’entrainement](#1-visualisation-des-données-dentrainement)
   - 2.2 [Visualisation des données de test](#2-visualisation-des-données-de-test)
3. [Modèle from scratch](#iii-modèle-frome-scratch)
   - 3.1 [Comparaison des ANN](#1-comparaison-des-ann)
     - 3.1.1 [Comparaison entre le Modèle ANN Simple sans et avec data augmentation](#a-comparaison-entre-le-modèle-ann-simple-sans-et-avec-data-augmentation)
     - 3.1.2 [Comparaison ANN complexe avec et sans data augmentation](#b-comparaison-ann-complexe-avec-et-sans-data-augmentation)
     - 3.1.3 [Quel est le meilleur modèle entre les ANN](#c-quel-est-le-meilleur-modèle-entre-les-ann)
   - 3.2 [Comparaison des CNN](#2-compraison-des-cnn)
     - 3.2.1 [Comparaison entre CNN simple sans et avec data augmentation](#a-comparaison-entre-cnn-simple-sans-et-avec-data-augmentation)
     - 3.2.2 [Comparaison des CNN complexe sans et avec data augmentation](#b-comparaison-des-cnn-complexe-sans-et-avec-data-augmentation)
     - 3.2.3 [Quel est le meilleur modèle entre les CNN](#c-quel-est-le-meilleur-modèle-entre-les-cnn)
4. [Transfer learning](#iv-transfer-learning)
   - 4.1 [Transfer learning avec VGG (frozen base) avec et sans data augmentation](#1-transfer-learning-avec-vgg-frozen-base-avec-et-sans-data-augmentation)
     - 4.1.1 [Les points importantes dans le code](#a-les-points-importantes-dans-le-code)
     - 4.1.2 [Comparaison](#b-comparaison)
   - 4.2 [Transfer learning avec VGG (unfrozen base) avec et sans data augmentation](#2-transfer-learning-avec-vgg-unfrozen-base-avec-et-sans-data-augmentation)
     - 4.2.1 [Code](#a-code)
     - 4.2.2 [Description du code](#b-description-du-code)
     - 4.2.3 [Comparaison](#c-comparaison)
   - 4.3 [Transfer Learning using ResNet50 avec et sans data augmentation](#3-transfer-learning-using-resnet50-avec-et-sans-data-augmentation)
     - 4.3.1 [Code](#a-code)
     - 4.3.2 [Comparaison](#b-comparaison)
     - 4.3.3 [Choix](#c-choix)
   - 4.4 [Quel est le meilleur modèle pour transfert learning](#4-quel-est-le-meilleur-modèle-pour-transfert-learning)
5. [Comparaison des modèles](#v-comparaison-des-modèles)
   - 5.1 [Pour le cas sans augmentation de données](#1-pour-le-cas-sans-augmentation-de-données)
   - 5.2 [Pour le cas avec augmentation de données](#2-pour-le-cas-avec-augmentation-de-données)

<fill the 

Certainly! Here's a structure for your chapters. You can fill in the actual content under each section.

## Contents
1. [Présentation de la base](#i-présentation-de-la-base)
2. [Visualiser quelques données](#ii-visualiser-quelques-données)
   - 2.1 [Visualisation des données d’entrainement](#1-visualisation-des-données-dentrainement)
   - 2.2 [Visualisation des données de test](#2-visualisation-des-données-de-test)
3. [Modèle from scratch](#iii-modèle-frome-scratch)
   - 3.1 [Comparaison des ANN](#1-comparaison-des-ann)
     - 3.1.1 [Comparaison entre le Modèle ANN Simple sans et avec data augmentation](#a-comparaison-entre-le-modèle-ann-simple-sans-et-avec-data-augmentation)
     - 3.1.2 [Comparaison ANN complexe avec et sans data augmentation](#b-comparaison-ann-complexe-avec-et-sans-data-augmentation)
     - 3.1.3 [Quel est le meilleur modèle entre les ANN](#c-quel-est-le-meilleur-modèle-entre-les-ann)
   - 3.2 [Comparaison des CNN](#2-compraison-des-cnn)
     - 3.2.1 [Comparaison entre CNN simple sans et avec data augmentation](#a-comparaison-entre-cnn-simple-sans-et-avec-data-augmentation)
     - 3.2.2 [Comparaison des CNN complexe sans et avec data augmentation](#b-comparaison-des-cnn-complexe-sans-et-avec-data-augmentation)
     - 3.2.3 [Quel est le meilleur modèle entre les CNN](#c-quel-est-le-meilleur-modèle-entre-les-cnn)
4. [Transfer learning](#iv-transfer-learning)
   - 4.1 [Transfer learning avec VGG (frozen base) avec et sans data augmentation](#1-transfer-learning-avec-vgg-frozen-base-avec-et-sans-data-augmentation)
     - 4.1.1 [Les points importantes dans le code](#a-les-points-importantes-dans-le-code)
     - 4.1.2 [Comparaison](#b-comparaison)
   - 4.2 [Transfer learning avec VGG (unfrozen base) avec et sans data augmentation](#2-transfer-learning-avec-vgg-unfrozen-base-avec-et-sans-data-augmentation)
     - 4.2.1 [Code](#a-code)
     - 4.2.2 [Description du code](#b-description-du-code)
     - 4.2.3 [Comparaison](#c-comparaison)
   - 4.3 [Transfer Learning using ResNet50 avec et sans data augmentation](#3-transfer-learning-using-resnet50-avec-et-sans-data-augmentation)
     - 4.3.1 [Code](#a-code)
     - 4.3.2 [Comparaison](#b-comparaison)
     - 4.3.3 [Choix](#c-choix)
   - 4.4 [Quel est le meilleur modèle pour transfert learning](#4-quel-est-le-meilleur-modèle-pour-transfert-learning)
5. [Comparaison des modèles](#v-comparaison-des-modèles)
   - 5.1 [Pour le cas sans augmentation de données](#1-pour-le-cas-sans-augmentation-de-données)
   - 5.2 [Pour le cas avec augmentation de données](#2-pour-le-cas-avec-augmentation-de-données)

## Présentation de la base
test
## Visualiser quelques données
test
## Modèle from scratch
## Transfer learning
## Comparaison des modèles