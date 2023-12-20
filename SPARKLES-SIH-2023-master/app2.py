from flask import Flask, render_template, request, redirect, url_for
import instaloader
import numpy as np
import tensorflow as tf

app = Flask(__name__)

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

# Load the trained model
load_model = tf.keras.models.load_model('trainedmodel')

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/advice')
def advice():
    # Render the advice.html template
    return render_template('advice.html')

@app.route('/about')
def about():
    # Render the about.html template
    return render_template('about.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect(url_for('predict'))
    return render_template('index.html')

# Flask route for rendering result.html
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
        selected_reasons = request.form.getlist('reasons')
        others = request.form.get('others')

        # Get Instagram data
        insta_data = get_instagram_data(username)

        if insta_data:
            # Convert Instagram data to NumPy array
            X_new = np.array([list(insta_data.values())], dtype=np.float32)

            # Make predictions
            predictions = load_model.predict(X_new)

            # Determine the result
            confidence_percentage = (1 - predictions[0][0]) * 100  # Subtract probability from 1 and multiply by 100

            # Determine the level of suspicion and behavior analysis
            level_of_suspicion = determine_level_of_suspicion(selected_reasons)

            # Create a dictionary mapping reasons to behavior messages
            behavior_messages = {reason: get_behavior_message(reason) for reason in selected_reasons if reason in ['spamMessages', 'linkSpamming', 'identityFraud', 'contenttheft', 'jobscam', 'fakenews', 'adfraud', 'bankAccount', 'otp', 'others']}

            # Redirect to the result page with the result as parameters
            return redirect(url_for('result', username=username, confidence=confidence_percentage,
                                    behavioral_analysis=level_of_suspicion, behavior_messages=behavior_messages))
        else:
            return render_template('result.html', username='N/A', confidence='N/A', behavioral_analysis='Profile not found.')
    except Exception as e:
        return render_template('result.html', username='N/A', confidence='N/A', behavioral_analysis=f"An error occurred: {str(e)}")

def get_behavior_message(reason):
    # Mapping of reasons to behavior messages
    behavior_mapping = {
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
    return behavior_mapping.get(reason, 'Normal behavious')


def determine_level_of_suspicion(selected_reasons):
    # Mapping of reasons to suspicion levels
    suspicion_mapping = {
        'spamMessages': 'Less',
        'linkSpamming': 'Abnormal',
        'identityFraud': 'Highly Suspicious',
        'contenttheft': 'Highly Suspicious',
        'jobscam': 'Abnormal',
        'fakenews': 'Suspicious',
        'adfraud': 'Abnormal',
        'bankAccount': 'Very Highly Suspicious',
        'otp': 'Very Highly Suspicious',
        'others': 'Suspicious',
    }

    # Determine suspicion level based on selected reasons
    suspicion_levels = [suspicion_mapping.get(reason, 'Normal') for reason in selected_reasons]

    # Return the highest suspicion level
    return max(suspicion_levels, key=lambda x: suspicion_levels.count(x))

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5004)
