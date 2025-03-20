import os
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
from PIL import Image
from models import db, User

app = Flask(__name__)

# Set secret key for sessions
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-123')

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Use PostgreSQL URL from environment variable if available, fallback to SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///leaderboard.db')
db.init_app(app)

with app.app_context():
    db.create_all()

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))  # Resize to match model input size
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Define a simple CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification (0 = male, 1 = female)
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load the model
try:
    model = tf.keras.models.load_model('gender_classification_model.h5')
except:
    print("No pre-trained model found. Creating a new model...")
    model = build_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Save the file
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)
        
        # Process the image and get prediction
        image = load_and_preprocess_image(image_path)
        prediction = model.predict(image)
        gender = "Female" if prediction[0][0] > 0.5 else "Male"
        confidence = float(abs(prediction[0][0] - 0.5) * 2)  # Convert to percentage
        
        return jsonify({
            'prediction': gender,
            'confidence': f"{confidence:.2%}",
            'image_path': image_path
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/leaderboard')
def leaderboard():
    try:
        users = User.query.order_by(User.score.desc()).limit(10).all()
        return render_template('leaderboard.html', leaderboard=users)
    except Exception as e:
        app.logger.error(f'Error accessing leaderboard: {str(e)}')
        return render_template('leaderboard.html', leaderboard=[], error='Unable to load leaderboard data')

@app.route('/quiz_result', methods=['POST'])
def quiz_result():
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()
    if not user:
        user = User(username=data['username'], score=0)
        db.session.add(user)
    if user.score is None:
        user.score = 0
    user.score += data['score']
    db.session.commit()
    return {'message': 'Score updated successfully'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port) 