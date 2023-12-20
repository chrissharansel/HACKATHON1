# ...

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
            if others:
                behavior = others  # Use "others" as behavior if provided
            elif reasons:
                behavior = reasons_to_behaviors.get(reasons[0], 'Suspicious behavior.')  # Use the behavior corresponding to the selected reason
            else:
                behavior = 'Suspicious behavior.'  # Default behavior if no reason or others are selected

            # Redirect to the result page with the result as parameters
            return redirect(url_for('result', username=username, prediction_result=prediction_result, confidence=confidence_percentage, behavioral_analysis=behavior))
        else:
            return render_template('result.html', username='N/A', confidence='N/A', behavioral_analysis='Profile not found.')
    except Exception as e:
        return render_template('result.html', username='N/A', confidence='N/A', behavioral_analysis=f"An error occurred: {str(e)}")

# ...
