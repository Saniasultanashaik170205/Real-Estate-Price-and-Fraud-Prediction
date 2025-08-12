from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pickle
import numpy as np
import pymysql
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load ML models
with open('house_price_model.pkl', 'rb') as f:
    pipeline, MODEL_FEATURES = pickle.load(f)

with open('fraud_model.pkl', 'rb') as f:
    fraud_pipeline, FRAUD_FEATURES = pickle.load(f)

# MySQL connection
db = pymysql.connect(host='localhost', user='root', password='', database='hybrid_db')
cursor = db.cursor()

@app.route('/')
def home():
    return render_template('open.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')
# Updated signup route to match your HTML form fields
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        # Check if email already exists
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        if cursor.fetchone():
            flash("Email already exists! Please login.", "error")
            return redirect(url_for('signup'))
        # Insert new user
        cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, password))
        db.commit()
        flash("Signup successful! Please login.", "success")
        return redirect(url_for('login'))
    return render_template('signup.html')

# Updated login route to match your HTML form fields
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        cursor.execute("SELECT * FROM users WHERE email=%s AND password=%s", (email, password))
        user = cursor.fetchone()
        if user:
            session['user'] = email
            return redirect(url_for('set_page'))
        else:
            flash("Invalid email or password", "error")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/set')
def set_page():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('set.html', user=session['user'])

@app.route('/index')
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', user=session['user'])

@app.route('/fraud')
def fraud():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('fraud.html', user=session['user'])

@app.route('/get_price', methods=['POST'])
def get_price():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        data = request.get_json()
        print("Received data:", data)  # Debug print
        living_area = float(data.get('living area', 0))
        basement_area = float(data.get('Area of the basement', 0))
        built_year = int(data.get('Built Year', 2000))
        renovation_year = int(data.get('Renovation Year', 0))
        input_data = {
            'number of bedrooms': float(data.get('number of bedrooms', 0)),
            'number of bathrooms': float(data.get('number of bathrooms', 0)),
            'living area': living_area,
            'lot area': float(data.get('lot area', 0)),
            'waterfront present': int(data.get('waterfront present', 0)),
            'grade of the house': int(data.get('grade of the house', 7)),
            'total_area': living_area + basement_area,
            'property_age': 2023 - built_year,
            'renovated': int(renovation_year > 0),
            'Number of schools nearby': int(data.get('Number of schools nearby', 5)),
            'Distance from the airport': float(data.get('Distance from the airport', 25))
        }
        print("Input data for model:", input_data)  # Debug print
        input_df = pd.DataFrame({feature: [input_data.get(feature, 0)] for feature in MODEL_FEATURES})
        print("Input DataFrame:", input_df)  # Debug print
        prediction = pipeline.predict(input_df)[0]
        return jsonify({'price': f"Rs {round(prediction, 2):,}", 'status': 'success'})
    except Exception as e:
        print("Error in /get_price:", e)  # Debug print
        return jsonify({'error': str(e), 'status': 'error'}), 400
    
@app.route('/get_fraud_score', methods=['POST'])
def get_fraud_score():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        data = request.get_json()
        input_data = {feature: data.get(feature, 0) for feature in FRAUD_FEATURES}
        input_df = pd.DataFrame([input_data])
        fraud_score = fraud_pipeline.predict(input_df)[0]
        return jsonify({'status': 'success', 'fraud_score': f"{fraud_score:.2f}"})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 400

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)
