
# House Price Prediction 

Predicting house sale prices in Ames, Iowa, finding what drives price, and suggesting houses to buyers.

## Project overview
- **Data:** 1,460 houses, 79 features describing almost every aspect of a home. Target: `SalePrice`.
- **Tasks:**
  1. Data analysis report (missing values, outliers, distributions, correlations, price trends)
  2. Prediction model and feature-price relationships
  3. Buyer suggestions by area, price and requirements
- **Extra reports:** Model Comparison Report and Report on Challenges Faced (both inside the notebook).

## Approach
1. **Cleaning:** missing values filled by meaning ("None" for absent features, GarageYrBlt filled with YearBuilt, LotFrontage by neighbourhood median); 2 abnormal outliers removed.
2. **Feature engineering:** TotalSF, HouseAge, RemodAge, TotalBath and Has... flags.
3. **Preprocessing:** log transform of the target and skewed features; ordinal and one-hot encoding (242 features in total).
4. **Modelling:** compared Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting and XGBoost with 5-fold cross-validation, then tuned the top three with GridSearchCV.
5. **Final model:** tuned Lasso (alpha = 0.003) inside a scikit-learn Pipeline, chosen on CV RMSE, with the test set used only for final reporting.

## Results
| Metric | Value |
|---|---|
| CV RMSE (log price) | 0.1176 |
| Test R² | 0.9155 |
| Test RMSE (log price) | 0.1193 |
| Average error (MAE) | about USD 14,218 |

The three tuned models (Lasso, Gradient Boosting, XGBoost) are within about 0.002 RMSE of each other. Lasso was chosen for its best CV score, speed and interpretability; Gradient Boosting is a good backup.

## Key findings
- OverallQual, total living area (TotalSF / GrLivArea), exterior and kitchen quality, bathrooms and neighbourhood raise price the most; house age lowers it.
- Lasso removed 140 of the 242 features, so a smaller set explains most of the price variation.
- Partial and abnormal sales behave differently from normal sales, so buyer recommendations use Normal sales only.
- Lowest price per sq ft: MeadowV, IDOTRR, OldTown. Premium areas: NridgHt, StoneBr, Somerst.

## Buyer suggestions
Recommendations rank houses by a cross-validated "value gap" (model estimate minus actual price) within a buyer's budget, size and quality limits. The model can't see a house's condition, so inspect before buying.

## Limitations
- Trained on Ames, Iowa data from 2006–2010 only (includes the housing downturn), so it may not transfer to other cities or today's market.
- Average error is about USD 14,000.

## How to run
1. Install: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter`
2. Place the dataset at `PRCP-1020-HousePricePred/Data/data.csv`, in the same folder as the notebook.
3. Open `PRCP-1020-HousePricePred.ipynb` and choose **Kernel → Restart & Run All**.

## Files
- `PRCP-1020-HousePricePred.ipynb`: full analysis, models and reports
- `README.md`: this file

## Dataset
Ames Housing data, provided with the project statement (PRCP-1020).
