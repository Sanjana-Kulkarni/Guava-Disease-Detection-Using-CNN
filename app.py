import os
import cv2
import sqlite3
import bcrypt
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the model
model = load_model("model/vgg_weights.hdf5")

labels = ['Disease Free', 'Phytopthora', 'Red rust', 'Scab', 'Styler and Root']

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}")
    image = cv2.resize(image, (32, 32))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    if 'user_id' in session:
        return render_template('test.html')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        try:
            conn = get_db_connection()
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                         (username, hashed_password))
            conn.commit()
            conn.close()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists. Try a different one.', 'danger')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            session['user_id'] = user['id']
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized access'}), 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        try:
            processed_image = preprocess_image(file_path)
            prediction = model.predict(processed_image)
            predicted_class = labels[np.argmax(prediction)]
            confidence = float(np.max(prediction) * 100)

            os.remove(file_path)  # Remove the file after prediction

            return jsonify({
                'predicted_class': predicted_class,
                'confidence': f'{confidence:.2f}%'
            }), 200

        except Exception as e:
            os.remove(file_path)  # Remove the file if an error occurs
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True)
