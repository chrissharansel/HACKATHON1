


from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import instaloader
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'  # SQLite database
db = SQLAlchemy(app)

# Define a mapping of reasons to behaviors
reasons_to_behaviors = {
    'spamMessages': 'Suspicious behavior.',
    'linkSpamming': 'Highly Abnormal behavior.',
    'identityFraud': 'Highly Suspicious behavior.',
    'contenttheft': 'Highly Suspicious behavior.',
    'jobscam': 'Suspicious behavior.',
    'fakenews': 'Highly Suspicious behavior.',
    'adfraud': 'Suspicious behavior.',
    'bankAccount': 'Very highly suspicious behavior.',
    'otp': 'Very highly suspicious behavior.',
    'others': 'Suspicious behavior.',
}

# Model loading
load_model = tf.keras.models.load_model('trainedmodel')

# Define the database model
class PredictionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    prediction_result = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    behavior = db.Column(db.String(100), nullable=False)

# Function to get Instagram data using Instaloader
def get_instagram_data(username):
    loader = instaloader.Instaloader()
    try:
        profile = instaloader.Profile.from_username(loader.context, username)
        return {
            "userFollowerCount": profile.followers,
            "userFollowingCount": profile.followees,
            "userBiographyLength": len(profile.biography),
            "userMediaCount": profile.mediacount,
            "userHasProfilPic": int(not profile.is_private and profile.profile_pic_url is not None),
            "userIsPrivate": int(profile.is_private),
            "usernameDigitCount": sum(c.isdigit() for c in profile.username),
            "usernameLength": len(profile.username),
        }
    except instaloader.exceptions.ProfileNotExistsException:
        print(f"Profile with username '{username}' not found.")
        return None

@app.route('/')
def main():
    return render_template('main.html')

# ... (your existing routes)
@app.route('/advice')
def advice():
    # Render the advice.html template
    return render_template('advice.html')

@app.route('/about')
def about():
    # Render the advice.html template
    return render_template('about.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect(url_for('predict'))
    return render_template('index.html')


@app.route('/result')
def result():
    # Get parameters from the query string or use default values
    username = request.args.get('username', 'N/A')
    confidence = float(request.args.get('confidence', 'N/A'))  # Ensure confidence is a float
    behavioral_analysis = request.args.get('behavioral_analysis', 'N/A')

    return render_template('result.html', username=username, confidence=confidence, behavioral_analysis=behavioral_analysis)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the username from the form submission
        username = request.form['username']
        reasons = request.form.getlist('reasons')
        others = request.form.get('others')

        # Get Instagram data
        insta_data = get_instagram_data(username)

        if insta_data:
            # Convert Instagram data to NumPy array
            X_new = np.array([list(insta_data.values())], dtype=np.float32)

            # Make predictions
            predictions = load_model.predict(X_new)

            # Get the number of checkboxes selected
            num_checkboxes_selected = len(reasons) + (1 if others else 0)

            # Perform behavioral analysis
            behavioral_analysis_result = "Behavioral Analysis: "
            if num_checkboxes_selected > 5:
                behavioral_analysis_result += "The user exhibits suspicious behavior."
            else:
                behavioral_analysis_result += "The user's behavior seems normal."

            # Determine the result
            prediction_result = 'Fake' if predictions[0][0] >= 0.5 else 'Real'
            confidence_percentage = (1 - predictions[0][0]) * 100  # Subtract probability from 1 and multiply by 100

            # Get the corresponding behavior based on the prediction result
            behavior = others if others else reasons_to_behaviors.get(reasons[0], 'Suspicious behavior.')  # Handle "others" separately

            # Save the prediction result to the database
            result_entry = PredictionResult(username=username, prediction_result=prediction_result, confidence=confidence_percentage, behavior=behavior)
            db.session.add(result_entry)
            db.session.commit()

            # Redirect to the result page with the result as parameters
            return redirect(url_for('result', username=username, prediction_result=prediction_result, confidence=confidence_percentage, behavioral_analysis=behavior))
        else:
            return render_template('result.html', username='N/A', confidence='N/A', behavioral_analysis='Profile not found.')
    except Exception as e:
        return render_template('result.html', username='N/A', confidence='N/A', behavioral_analysis=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    db.create_all()  # Create database tables before running the app
    app.run(debug=True, host='127.0.0.1', port=5004)
