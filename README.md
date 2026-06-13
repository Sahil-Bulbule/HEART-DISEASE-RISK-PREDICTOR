# Heart Disease Risk Predictor

A professional Streamlit application for cardiovascular risk assessment using a Support Vector Machine (SVM) model.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place your model files** in the same folder as `app.py`:
   - `SVM2_Spam.pkl` — trained SVM model
   - `scaler2.pkl` — fitted StandardScaler
   - `columns2.pkl` — list/Index of training column names (after one-hot encoding)

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## How the prediction works

1. User fills in the form (Age, BP, Cholesterol, etc.)
2. Categorical fields (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope) are one-hot encoded with `pd.get_dummies`
3. The encoded dataframe is **re-indexed** to exactly match `columns2.pkl` (missing columns filled with 0, extra columns dropped)
4. The aligned dataframe is scaled with `scaler2.pkl`
5. `SVM2_Spam.pkl` predicts 0 or 1
   - **0 → Low Risk — Healthy Parameters** (green card)
   - **1 → High Cardiovascular Risk Detected** (red card)

## Test values

| Parameter | Low-Risk | High-Risk |
|-----------|----------|-----------|
| Age | 25 | 70 |
| RestingBP | 110 | 180 |
| Cholesterol | 150 | 350 |
| FastingBS | 0 | 1 |
| MaxHR | 180 | 90 |
| Oldpeak | 0.0 | 4.0 |
| Sex | Female | Male |
| ChestPainType | ATA | ASY |
| RestingECG | Normal | ST |
| ExerciseAngina | N | Y |
| ST_Slope | Up | Down |

## Debugging

Expand the **Debug Info** section after each prediction to inspect:
- Raw model prediction value
- Input dataframe before encoding
- One-hot encoded dataframe
- Aligned dataframe (matched to training columns)
- Scaled array shape
- Column mismatch report (columns added / dropped)
