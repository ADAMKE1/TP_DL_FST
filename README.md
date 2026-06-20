# Deep Learning — Projet de fin de module

Quatre travaux pratiques couvrant les principales familles de modèles de deep learning,
de la classification d'images au traitement du langage naturel.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-TP1%20%7C%20TP2%20%7C%20TP4-ee4c2c)
![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-TP3-ff6f00)
![Colab](https://img.shields.io/badge/Exécuter%20sur-Google%20Colab-f9ab00)

---

## Vue d'ensemble

Chaque notebook constitue un pipeline complet et autonome
(**chargement des données → prétraitement → modèle → entraînement → évaluation**).
Ensemble, ils forment une progression logique :
**images** (CNN) → **séries temporelles** (LSTM) → **régression tabulaire** (LSTM) → **langage** (Transformer).
Le **transfer learning / fine-tuning** est un fil conducteur commun aux TP1 et TP4.

| TP | Tâche | Modèle | Jeu de données | Résultat clé |
|----|-------|--------|----------------|--------------|
| **TP1** | Classification d'images | CNN personnalisé, DenseNet, transfer learning (AlexNet / ResNet18 / VGG16) | FashionMNIST | Précision **92,3 %** (CNN), **90,1 %** (ResNet18 avec fine-tuning progressif) |
| **TP2** | Prévision de séries temporelles | LSTM empilé (3 couches) | UCI Household Electric Power | R² **0,61**, MAE **0,32 kW** |
| **TP3** | Régression (+ TinyML) | LSTM | Consommation de carburant | R² **0,95**, MAE **0,41 L/100 km** |
| **TP4** | Classification de texte | DistilBERT (fine-tuning) | emotion (6 classes) | Précision **91,9 %**, F1 pondéré **92,0 %** |

---

## Structure du dépôt

```
.
├── tp1_fashionmnist.ipynb     # CNN + DenseNet + transfer learning
├── tp2_lstm.ipynb             # LSTM empilé — consommation électrique
├── tp3_fuel_lstm.ipynb        # Régression LSTM — consommation carburant (TinyML)
├── tp4_transformers.ipynb     # Fine-tuning DistilBERT — classification d'émotions
└── README.md
```

---

## Les quatre travaux pratiques

### TP1 — FashionMNIST (CNN, DenseNet, Transfer Learning)

Classification de 10 catégories de vêtements. Un CNN personnalisé est d'abord entraîné
from scratch, puis des modèles pré-entraînés sur ImageNet (DenseNet121, AlexNet, ResNet18,
VGG16) sont adaptés aux images en niveaux de gris 28×28 (conversion en 3 canaux,
redimensionnement à 224×224, normalisation ImageNet, nouvelle tête de classification à
10 classes). Différentes **stratégies de gel des couches** sont comparées : extraction de
features, gel partiel et fine-tuning progressif.

### TP2 — Consommation électrique (LSTM empilé)

Prévision univariée pas-à-pas de la puissance active d'un foyer. La série temporelle à
la minute est rééchantillonnée à l'**heure**, des caractéristiques temporelles cycliques
sont construites, puis un LSTM à 3 couches empilées est entraîné. L'évaluation repose
sur le MAE, le RMSE et le R².

### TP3 — Consommation de carburant (LSTM, TinyML)

Prédiction de la consommation de carburant à partir des caractéristiques d'un véhicule,
via un LSTM très léger (~2 300 paramètres). Inclut une analyse exploratoire (pairplot,
corrélation de Spearman), une analyse des résidus, et une **exportation TinyML optionnelle**
en code C pour déploiement sur microcontrôleur.

### TP4 — Classification d'émotions (Transformers)

Fine-tuning de **DistilBERT** sur le dataset `emotion` (tristesse, joie, amour, colère,
peur, surprise). Illustre le paradigme **pré-entraînement → fine-tuning** en NLP, avec
évaluation par précision, F1 pondéré et matrice de confusion.

---

## Exécution des notebooks

Tous les notebooks sont conçus pour **Google Colab** (ou Kaggle) et installent leurs
dépendances automatiquement dans les premières cellules.

1. Ouvrir le notebook dans Colab.
2. Pour TP1 et TP4, activer un GPU : **Exécution → Modifier le type d'exécution → GPU T4**.
3. Exécuter toutes les cellules de haut en bas.

| Notebook | GPU | Accès aux données |
|----------|-----|-------------------|
| TP1 | Recommandé | Automatique (torchvision télécharge FashionMNIST) |
| TP2 | Optionnel | Automatique (téléchargement depuis UCI) |
| TP3 | Non nécessaire | **Charger `FuelConsumption.csv`** dans le répertoire de travail (Kaggle : `krupadharamshi/fuelconsumption`) |
| TP4 | Recommandé | Automatique (Hugging Face `datasets`) |

> **Astuce** — Pour ouvrir directement dans Colab :
> ```
> https://colab.research.google.com/github/abdellah-darni/Deep-Lerning-proj/blob/main/tp1_fashionmnist.ipynb
> ```

---

## Stack technique

| Bibliothèque | Utilisation |
|---|---|
| **PyTorch** | TP1, TP2, TP4 |
| **TensorFlow / Keras** | TP3 (conservé en TF pour l'export TinyML) |
| **Hugging Face** `transformers` + `datasets` | TP4 |
| scikit-learn, pandas, matplotlib, seaborn | Prétraitement, métriques, visualisation |

---

## Remarques

- Le notebook **TP4** utilise un `collate_fn` personnalisé plutôt que
  `datasets.set_format("torch")`, afin de contourner une incompatibilité connue entre
  `datasets` et `torchvision` sur Colab.
- Dans **TP2**, le scaler est ajusté uniquement sur le jeu d'entraînement, après
  rééchantillonnage horaire, afin d'éviter toute fuite de données.
- Le R² élevé de **TP3** s'explique en grande partie par la corrélation quasi-parfaite
  (~0,99) entre la consommation de carburant et les émissions de CO₂, utilisées comme
  variable d'entrée — voir la discussion dans le notebook.
