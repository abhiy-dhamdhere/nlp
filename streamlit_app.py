import numpy as np
import pickle
import streamlit as st

# Load the model and label encoder
with open("model_pipeline.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Define the mapping for the dependent variable
def map_prediction(prediction):
    return label_encoder.inverse_transform(prediction)[0]

def news_classification(news_headline):
    prediction = classifier.predict([news_headline])
    return map_prediction(prediction)

def main():
    # Set the page title and favicon
    st.set_page_config(page_title="News Classification", page_icon="📰")

    # Custom header styling
    st.markdown(
        """
        <style>
        .header {
            background-color: #4CAF50; 
            padding: 10px; 
            border-radius: 10px; 
            text-align: center;
        }
        .header h2 {
            color: white; 
            font-family: Arial, sans-serif;
        }
        </style>
        <div class="header">
            <h2>News Classification</h2>
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.write("")

    st.write(
        """
        Welcome to the News Classification App! 
        Simply enter a news headline below and click "Predict" to see its category.
        """
    )

    # Text input for news description
    news_description = st.text_input("Enter News Description", help="Type the news headline you want to classify.")

    # Button to trigger prediction
    if st.button("Predict", help="Click to classify the news headline"):
        if news_description.strip():
            result = news_classification(news_description)
            st.success(f'The category is: **{result}**')
        else:
            st.error("Please enter a news description.")

    # Footer
    st.markdown(
        """
        <hr style="border-top: 2px solid #4CAF50;">
        <footer style="text-align: center;">
            <p>Developed by Abhishek Dhamdhere &copy; 2024</p>
        </footer>
        """, 
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
