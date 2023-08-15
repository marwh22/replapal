import pandas as pd
import re
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import webbrowser

# Load the data
data = pd.read_csv("C:/Users/HP/Downloads/rephrased_text1e2.csv")

# Drop any rows with missing values
data = data.dropna()

# Clean the Complaint column by converting all text to lowercase
data['Complaints'] = data['Complaints'].str.lower()
data['problem'] = data['problem'].str.lower()

# Convert the brand and model columns to lowercase
data['brand'] = data['brand'].str.lower()
data['model'] = data['model'].str.lower()

# Clean the problems column by removing any non-alphanumeric characters
data['problem'] = data['problem'].astype(str).apply(lambda x: re.sub(r'\W+', ' ', x))

# Remove any leading or trailing whitespaces in all columns:
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Convert the year column to integer
data['year'] = data['year'].astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['Complaints'], data['problem'], test_size=0.2, random_state=42)

# Define the pipeline to vectorize the text and train a Naive Bayes classifier
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB())
])

# Train the model on the training set
text_clf.fit(X_train, y_train)

# Get user input using Streamlit widgets
brand = st.selectbox('Enter the car brand:', data['brand'].unique())
year = st.number_input('Enter the car year:', value=2021, min_value=1900, max_value=2023)
model = st.selectbox('Enter the car model:', data[data['brand'] == brand]['model'].unique())

# Test the model on the testing set
predicted = text_clf.predict(X_test)

# Calculate the accuracy score
accuracy = (predicted == y_test).mean()

def predict_problems(complaint):
    predicted_probs = text_clf.predict_proba([complaint])[0]
    top_5_indices = predicted_probs.argsort()[-5:][::-1]
    top_5_probs = predicted_probs[top_5_indices]
    top_5_problems = text_clf.classes_[top_5_indices]

    # Get the prices for the top 5 problems
    prices = []
    for problem in top_5_problems:
        price = data.loc[data['problem'] == problem, 'price'].values[0]
        prices.append(price)

    # Return the top 5 problems, their associated probabilities, and estimated costs
    return [(problem, prob, price) for problem, prob, price in zip(top_5_problems, top_5_probs, prices)]


st.title('The Complaint ')
complaint = st.text_input('Please Enter Your car Complaint')
if not complaint:
    st.write('You must write the complaint')
else:
    problems = predict_problems(complaint.lower())
    st.write('Top 5 possible problems:')
    st.markdown('<div style="display:inline-block;width:40%"> <b>Problem</b> </div>'
                '<div style="display:inline-block;width:30%"> <b>Probability</b> </div>'
                '<div style="display:inline-block;width:30%"> <b>Cost</b> </div>',
                unsafe_allow_html=True)
    for problem, prob, price in problems:
        st.markdown(f'<div style="background-color: white; color: black; padding: 10px; margin-bottom: 10px;">'
                    f'<div style="display:inline-block;width:40%"> {problem} </div>'
                    f'<div style="display:inline-block;width:30%"> {prob:.2f} </div>'
                    f'<div style="display:inline-block;width:30%"> {price:.2f} </div>'
                    f'</div>',
                    unsafe_allow_html=True)
        
# Add a button at the end of the page
import webbrowser
import platform

# Check the user's operating system
user_os = platform.system()

if st.button('car workshop'):
    # Open the URL based on the user's operating system
    url = 'https://www.google.com/maps/search/%D9%88%D8%B1%D8%B4%D8%A9+%D8%B3%D9%8A%D8%A7%D8%B1%D8%A9%E2%80%AD/@21.4392314,39.8167569,13z/data=!3m1!4b1?entry=ttu'
    
    if user_os == 'Darwin':  # macOS
        webbrowser.get('open -a /Applications/Google\ Chrome.app %s').open_new_tab(url)
    elif user_os == 'Windows':
        webbrowser.get('windows-default').open_new_tab(url)
    elif user_os == 'Linux':
        webbrowser.get('google-chrome').open_new_tab(url)
    else:
        webbrowser.open_new_tab(url)