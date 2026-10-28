# Great American Coffee Taste Test: Statistical Inference Project

Streamlit dashboard covering the SI lab practicals on a real survey dataset (~4,000 respondents).

**Dataset (Kaggle):** https://www.kaggle.com/datasets/datalab351/great-american-coffee-taste-test
Download the CSV and save it next to `app.py` as `coffee_survey.csv` (or upload it in the sidebar).
If no file is found, the app falls back to the TidyTuesday copy of the same survey.

## Run
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Practicals covered
| Tab | Practical | Methods |
|---|---|---|
| 1 Web Scraping | P1 | Requests, BeautifulSoup, DataFrame, duplicates, missing values, CSV |
| 2 Descriptive & Graphs | P2 | mean, median, mode, variance, SD, quartiles, IQR, describe(); histogram, box, bar, pie, scatter, line |
| 3 Distributions | P3 | Poisson, binomial, normal fits |
| 4 Sampling & CLT | P4 | simple random, systematic, stratified, cluster, with/without replacement, CLT |
| 5 CI & Tests | P5 | t-based and bootstrap CI, one-sample z-test and t-test |
| 6 Chi-Square | P6 | independence test, Cramér's V |
| 7 Regression | P7 | Linear, Ridge, Lasso, Decision Tree, Random Forest, live prediction |
| 8 Classification | P8 | Logistic, KNN, Decision Tree, Random Forest, SVM, confusion matrices |
| Bonus | - | Pearson correlation, one-way ANOVA |

## Notes
- Survey participants are self-selected, so results describe this sample, not all Americans.
- Column names follow the original survey (`age`, `cups`, `favorite`, `expertise`, `gender`, `prefer_overall`, `coffee_a_bitterness`, etc.). Features that are missing from a file are skipped.
