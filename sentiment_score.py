import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
import spacy

# Download NLTK stopwords (if not already downloaded)
nltk.download('stopwords')

# Download spaCy model for named entity recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Define a function to analyze sentiment and generate a word cloud
def analyze_sentiment_and_generate_wordcloud(text, text_col, custom_stopwords, max_words, relative_scaling, background_color, colormap):
    # Create a DataFrame from the input text
    df = pd.DataFrame({text_col: [text]})

    # Apply sentiment analysis to the specified text column and create new columns
    df['Sentiment Score'], df['Sentiment Label'] = zip(*df[text_col].apply(analyze_sentiment))

    # Apply stopword removal and NER
    df['Filtered Review'] = df[text_col].apply(remove_non_sentiment_words, custom_stopwords=custom_stopwords)

    # Join all filtered reviews into a single text
    all_filtered_reviews = ' '.join(df['Filtered Review'])

    # Define word cloud parameters
    wordcloud = WordCloud(width=800, height=400, background_color=background_color, colormap=colormap,
                          max_words=max_words, relative_scaling=relative_scaling)

    # Generate the word cloud
    wordcloud.generate(all_filtered_reviews)

    # Display the word cloud
    st.image(wordcloud.to_image())

    # Calculate KPIs
    sentiment_score = df['Sentiment Score'].mean()  # Calculate the average sentiment score

    # Display KPIs
    st.write("Average Sentiment Score: {:.2f}".format(sentiment_score))
    st.write("Sentiment scores range from -1 (most negative) to 1 (most positive), with 0 indicating neutrality.")

# Function to categorize sentiment
def categorize_sentiment(score):
    if score > 0.5:
        return 'Positive'
    elif score < -0.5:
        return 'Negative'
    else:
        return 'Neutral'

# Function to process and analyze the 'Review' column
def analyze_sentiment(text):
    # Tokenize the text
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    # Join the filtered words back into a sentence
    filtered_text = ' '.join(filtered_words)
    
    # Analyze sentiment using TextBlob
    sentiment = TextBlob(filtered_text).sentiment.polarity
    
    # Categorize sentiment
    sentiment_label = categorize_sentiment(sentiment)
    
    return sentiment, sentiment_label

# Function to remove stopwords and non-sentiment entities using spaCy NER
def remove_non_sentiment_words(text, custom_stopwords):
    doc = nlp(text)
    
    # Remove stopwords and non-sentiment entities (e.g., names, places, things)
    filtered_words = [token.text for token in doc if token.text.lower() not in custom_stopwords and token.ent_type_ != 'PERSON']
    
    # Join the filtered words back into a sentence
    filtered_text = ' '.join(filtered_words)
    
    return filtered_text

if __name__ == '__main__':
    st.title("Sentiment Analysis with Word Cloud")
    st.sidebar.title("Settings")

    # Initialize the text and text_col variables
    text = ""
    text_col = ""

    # Upload file or paste text
    input_option = st.sidebar.radio("Input Option", ["Upload File", "Paste Text"])
    
    if input_option == "Upload File":
        uploaded_file = st.sidebar.file_uploader("Upload a File", type=["csv", "xlsx", "txt", "pdf"])
        
        # Only read the file if it's uploaded
        if uploaded_file is not None:
            try:
                # Handle CSV files
                if uploaded_file.type == "application/vnd.ms-excel" or uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file)
                    if not df.empty:
                        text_col = st.sidebar.selectbox("Select Text Column", df.columns)
                        text = df[text_col].to_string(index=False)
                    else:
                        st.warning("The selected CSV file is empty.")
                # Handle Excel files
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                    df = pd.read_excel(uploaded_file)
                    if not df.empty:
                        text_col = st.sidebar.selectbox("Select Text Column", df.columns)
                        text = df[text_col].to_string(index=False)
                    else:
                        st.warning("The selected Excel file is empty.")
                else:
                    # Read the file directly if it's not a CSV or Excel file
                    text = uploaded_file.read()
            except Exception as e:
                st.error(f"An error occurred while processing the file: {str(e)}")
    else:
        # Allow users to paste text
        text = st.sidebar.text_area("Paste Text Here", "")

    if text:
        # Define custom stopwords
        custom_stopwords = st.sidebar.text_area("Custom Stopwords (comma-separated)", value="")

        # Word Cloud Settings
        max_words = st.sidebar.slider("Max Words", min_value=1, max_value=500, value=100)
        relative_scaling = st.sidebar.slider("Relative Scaling", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        background_color = st.sidebar.color_picker("Background Color", value="#ffffff")
        colormap = st.sidebar.selectbox("Colormap", plt.colormaps())

        custom_stopwords = set(custom_stopwords.split(','))

        try:
            analyze_sentiment_and_generate_wordcloud(text, text_col, custom_stopwords, max_words, relative_scaling, background_color, colormap)
        except Exception as e:
            st.error(f"Select text columns you need to analyze: {str(e)}")

