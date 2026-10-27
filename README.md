# Coffee Consumption Statistical Analysis

A Python Statistical Inference lab project with a Streamlit dashboard and a synthetic dataset.

## Files
- `coffee_data.csv`: 500 synthetic participant records for demonstration.
- `statistics_backend.py`: data cleaning, descriptive statistics, bootstrap confidence interval, CLT simulation, one-sample t-test, Pearson correlation, and one-way ANOVA.
- `app.py`: Streamlit dashboard.
- `requirements.txt`: Python dependencies.
- `Coffee_Consumption_Colab.ipynb`: notebook for running backend analyses in Google Colab.

## Dataset columns
- `Age`: participant age in years
- `Cups_Per_Day`: daily cups of coffee
- `Coffee_Type`: Espresso, Cappuccino, Latte, or Instant
- `Money_Per_Week`: estimated weekly spending in INR
- `Days_Per_Week`: days per week the participant consumes coffee
- `Consumption_Time`: usual consumption time

## Important
The data is synthetic, generated for educational demonstration. It is not real survey data and should not be reported as evidence about real people.

## Run the Streamlit dashboard locally
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Use in Google Colab
Upload the CSV and `statistics_backend.py` to Colab, or clone your GitHub repository. Then run the notebook `Coffee_Consumption_Colab.ipynb`. The notebook focuses on backend analysis and charts. Streamlit is a web app; running it from Colab requires a separate tunnel/hosting setup.

## Statistical methods
- Descriptive statistics
- Bootstrap confidence interval for the mean
- Sampling distribution / CLT simulation
- Two-sided one-sample t-test
- Pearson correlation
- One-way ANOVA

Interpret p-values alongside test assumptions and study design. Statistical association does not imply causation.
