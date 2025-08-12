# RealEstateWebsite

This repository contains a Real Estate web app (Flask) for house price prediction and fraud detection.
Files included:
- app.py
- database.py
- housepricing.py
- fraud.py
- house_price_model.pkl (placeholder)
- fraud_model.pkl (placeholder)
- House Price India.csv (dataset)
- fraud.csv (dataset)
- users.sql
- templates/ (HTML templates)

How to run (basic):
1. Install dependencies: `pip install flask pandas scikit-learn pymysql`
2. Place the CSV files in the project root (they are included).
3. (Optional) Train models by running `housepricing.py` and `fraud.py` to generate real .pkl files.
4. Run the app: `python app.py`
5. Open `http://127.0.0.1:5000/` in your browser.

Note: Update MySQL credentials in `app.py` or `database.py` if needed.
