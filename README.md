# TARDIS - Train Analysis and Retards Diagnostic Information System
<img alt="TARDIS Logo" src="https://upload.wikimedia.org/wikipedia/fr/thumb/a/a1/Logo_SNCF_%282011%29.svg/1200px-Logo_SNCF_%282011%29.svg.png" width="300" height="auto">

## Description

TARDIS est un tableau de bord interactif permettant d'analyser les retards des trains en France. L'application offre une visualisation complète des données historiques de ponctualité, avec des statistiques détaillées sur les causes de retard et les performances par ligne.

## Prérequis

- Python 3.7+
- Connexion Internet pour les cartes interactives

# Cloner le dépôt
`git clone git@github.com:EpitechPGEPromo2029/G-AIA-210-NAN-2-1-tardis-noah.savoye.git`

cd tardis

# Créer et activer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate - Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Créer les données

[![Ouvrir dans Jupyter](https://img.shields.io/badge/Jupyter-Ouvrir%20Notebook-orange?style=for-the-badge&logo=Jupyter)](tardis_eda.ipynb)

Pour explorer l'analyse exploratoire des données, vous pouvez cliquer sur le bouton ci-dessus ou exécuter la commande suivante:

```bash
jupyter notebook tardis_eda.ipynb
```

# Configuration

Assurez-vous que **les fichiers suivants sont présents dans le répertoire**:
- `cleaned_dataset.csv` - Données principales sur les trajets et retards
- `comments_dataset.csv` - Commentaires associés aux retards (optionnel)
- `list.csv` - Liste des gares avec coordonnées géographiques

# Utilisation

streamlit run dashboard.py

Allez à l'adresse:
- `http://localhost:8501`

----

## Dashboard

# 🇬🇧 SNCF Team – User Manual

## 📄 First Page

On the first page, you can:

- **Plan your route**  
  Select your departure and destination stations, then choose your desired travel period.

- **View the map**  
  See a visual representation of your selected route on the interactive map.

- **Navigate to the second page**  
  Click the **"Analyse"** button to access detailed data and visualizations.

---

## 📄 Second Page

On the second page, you can:

- **Select your route again**  
  Choose your departure and destination stations, along with your travel period.

- **Access detailed insights**:

  - 🔢 **View the train number(s)** for the selected month  
  - 📊 **"Distribution des retards (boxplot)"**  
    A boxplot showing delay distribution by delay type  
  - 📈 **"Nombre de trains retardés en fonction des mois"**  
    A line or bar chart showing the number of delayed trains per month  
  - 📉 **"Moyenne des pourcentages causes"**  
    A chart showing the average percentage of each delay cause  
  - 💬 **Comments list**  
    Browse user or system comments related to the selected route

- **Return to the first page**  
  Click the **"Analyse"** button again to go back

---

# 🇫🇷 Équipe SNCF – Manuel d’utilisation

## 📄 Première page

Sur la première page, vous pouvez :

- **Planifier votre trajet**  
  Sélectionnez votre station de départ et votre destination, puis choisissez la période souhaitée.

- **Afficher la carte**  
  Visualisez votre itinéraire sur une carte interactive.

- **Naviguer vers la page suivante**  
  Cliquez sur le bouton **"Analyse"** pour accéder aux données détaillées et aux graphiques.

---

## 📄 Deuxième page

Sur la deuxième page, vous pouvez :

- **Redéfinir votre trajet**  
  Choisissez à nouveau vos stations de départ et d’arrivée, ainsi que votre période de voyage.

- **Accéder à des données détaillées** :

  - 🔢 **Voir le(s) numéro(s) du train** pour le mois sélectionné  
  - 📊 **"Distribution des retards (boxplot)"**  
    Un diagramme en boîte montrant la répartition des retards selon leur type  
  - 📈 **"Nombre de trains retardés en fonction des mois"**  
    Un graphique affichant le nombre de trains en retard par mois  
  - 📉 **"Moyenne des pourcentages causes"**  
    Un graphique indiquant le pourcentage moyen de chaque cause de retard  
  - 💬 **Liste des commentaires**  
    Consultez les commentaires liés à l’itinéraire sélectionné

- **Retourner à la première page**  
  Cliquez de nouveau sur le bouton **"Analyse"**
