# 🌲 Tree Rot Prediction with Random Forest

This project predicts tree rot using environmental and categorical data. The dataset is enriched with weather data from [MET Norway's Frost API](https://frost.met.no/), and a Random Forest model is used to classify trees based on their risk of rot.

---

## 📁 Dataset

The dataset (`data.csv`) includes:

- `d_type`: Type of the tree
- `species`: Tree species
- `municipality`: Municipality name
- `date`: Date of measurement
- `rot`: Target label (`Yes`/`No`)
- `lat`, `long`: Coordinates
- `mean_temp_3m`, `mean_temp_1y`, `mean_temp_5y`: Temperature averages
- `min_temp`, `max_temp`: Temperature extremes
- `humidity`, `soil_humidity`: Environmental moisture indicators

---

## ⚙️ Data Preparation

The preprocessing pipeline includes:

- Handling missing values
- Normalizing numeric features
- One-hot encoding of categorical variables
- Fetching weather data using [Frost API](https://frost.met.no/)

---

## 🧠 Model

The main Random Forest model is implemented in [`models/random_forest.py`](models/random_forest.py). Features:

- Preprocessing with `ColumnTransformer`
- Balanced training/test sampling
- Evaluation with classification metrics

---

## 🚀 Getting Started

1. **Clone the repo**:
   ```bash
   git clone https://github.com/AndKlet/RotAnalysis
   cd RotAnalysis


2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt

3. **Run the model:**
    ```bash
    python models/random_forest.py