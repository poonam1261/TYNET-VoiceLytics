import os

import pymysql
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length
from flask_bcrypt import Bcrypt
from graph_creator import create_communication_graph
import re

app = Flask(__name__)
app.secret_key = 'your_secret_key'
bcrypt = Bcrypt(app)

# Configure MySQL connection
db = pymysql.connect(
    host="localhost",
    user="root",
    password="poonam@22cse1014",
    database="flasklogin"
)

# Mock user data
users = {'testuser': bcrypt.generate_password_hash('password123').decode('utf-8')}

app.config['GRAPH_FOLDER'] = 'static/graphs'

# Ensure the directory exists
os.makedirs(app.config['GRAPH_FOLDER'], exist_ok=True)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)



# Flask-WTF Login Form
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=3, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=6, max=25)])
    submit = SubmitField('Login')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        if account and bcrypt.check_password_hash(account['password'], password):
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            msg = 'Logged in successfully !'
            return redirect(url_for('voice_input'))
        else:
            msg = 'Incorrect username / password !'
    return render_template('login.html', msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        username = request.form['username']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        email = request.form['email']
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address !'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers !'
        elif not username or not password or not email:
            msg = 'Please fill out the form !'
        else:
            cursor.execute('INSERT INTO accounts (username, password, email) VALUES (%s, %s, %s)', (username, password, email))
            db.commit()
            msg = 'You have successfully registered !'
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('register.html', msg=msg)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/voice_input', methods=['GET', 'POST'])
def voice_input():
    if request.method == 'POST':
        if 'voice' in request.files:
            voice_file = request.files['voice']
            if voice_file.filename == '':
                return jsonify({'status': 'error', 'message': 'No file selected. Please upload a valid voice file.'}), 400
            if voice_file and allowed_file(voice_file.filename):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], voice_file.filename)
                voice_file.save(filepath)
                return jsonify({'status': 'success', 'message': f"File '{voice_file.filename}' uploaded successfully for analysis."}), 200
            else:
                return jsonify({'status': 'error', 'message': 'Invalid file type. Please upload a valid voice file.'}), 400
        else:
            return jsonify({'status': 'error', 'message': 'No file uploaded.'}), 400
    return render_template('voiceInput.html')

@app.route('/submit', methods=['POST'])
def submit_audio():
    # Check if the file part exists
    if 'voice' not in request.files:
        return jsonify({"error": "No file part"}), 400

    audio_file = request.files['voice']

    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file to the upload folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
    audio_file.save(file_path)

    # Perform further processing or analysis if needed
    print("successful")
    return render_template('result.html',)

@app.route('/test', methods=['POST'])
def test():
    graph_path = os.path.join(app.config['GRAPH_FOLDER'], 'communication_graph.png')

    # Generate the graph if it doesn't already exist
    if not os.path.exists(graph_path):
        create_communication_graph(graph_path)

    # Pass graph URL to the template
    graph_url = url_for('static', filename='graphs/communication_graph.png')
    return render_template('result.html', graph_url=graph_url)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/get-analysis-data', methods=["GET"])
def get_analysis_data():
    # Simulate analysis data (replace this with actual ML analysis results)
    analysis_result = {
        "labels": ["Emotion", "Pitch", "Grammar Score", "Communication Score"],
        "values": [8, 6, 7, 9]  # Example scores - replace with ML analysis results
    }

    # Send data as JSON to the front-end
    return jsonify(analysis_result)

@app.route('/get-emotion-data', methods=["GET"])
def get_emotion_data():
    # Simulated emotion analysis results (replace with ML results in production)
    emotion_data = {
        "labels": ["Happy", "Sad", "Angry", "Surprised", "Fearful", "Neutral"],
        "values": [30, 20, 15, 25, 5, 5]  # These values represent intensities of each emotion.
    }

    return jsonify(emotion_data)

if __name__ == '__main__':
    app.run(debug=True)
