from dotenv import find_dotenv, load_dotenv             # Import functions to load environment variables from a .env file.
from transformers import pipeline                       # Import pipeline from Hugging Face Transformers for building NLP models.
from langchain_groq import ChatGroq                     # Import ChatGroq, a chat-based model API for generating responses.
from langchain_core.prompts import PromptTemplate       # Import PromptTemplate for creating reusable prompt templates.
import requests                                         # Import requests library for making HTTP requests.
import os                                               # Import os module to access environment variables.
import streamlit as st                                  # Import Streamlit to create a web application.

# Load environment variables from a .env file
load_dotenv(find_dotenv())

# Load Hugging Face API key from environment variables
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

# Load GROQ API key from environment variables (though it's not used here)
os.getenv("GROQ_API_KEY")


# Function to convert an image to text
def imgtotext(url):  # Define a function that takes the path or URL of an image as input.
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")  # Initialize an image-to-text model.

    text = image_to_text(url)[0]['generated_text']  # Use the model to generate a caption for the image.

    print(text)  # Print the generated text to the console.
    return text  # Return the generated text.


# Class for generating a short story using an LLM
class Chain:
    def __init__(self):  # Initialize the Chain class.
        self.llm = ChatGroq(                            # Create an instance of the ChatGroq LLM model.
            temperature=0,                              # Set the temperature (controls randomness in responses).
            groq_api_key=os.getenv("GROQ_API_KEY"),     # Load the GROQ API key from environment variables.
            model="llama-3.3-70b-versatile"             # Specify the model version to use.
        )

    def write_story(self, scenario):  # Define a method to write a short story based on a scenario.
        prompt_email = PromptTemplate.from_template(
            """You are a storyteller;
            You should generate a short story base on a simple narrative, the story should be no more than 20 words;
    
            CONTEXT: {scenario}
            STORY: 
            
            """
        )  # Define a prompt template for the LLM to generate a short story.

        chain_email = prompt_email | self.llm                      # Combine the prompt template with the LLM to form a chain.
        res = chain_email.invoke({"scenario": str(scenario)})      # Invoke the chain with the given scenario.
        return res.content                                         # Return the generated story.


# Function to convert text to speech
def texttospeech(story):
    API_URL = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts"    # Set the API endpoint for text-to-speech.
    headers = {"Authorization": "Bearer {HUGGINGFACEHUB_API_TOKEN}"}                  # Set the authorization header with an API token.
    payloads = {
        "inputs": story  # Pass the story text as input to the API.
    }

    response = requests.post(API_URL, headers=headers, json=payloads)  # Send a POST request to the API.
    with open("audio.mp3", 'wb') as file:                              # Open a file to save the audio response.
        file.write(response.content)                                   # Write the audio content to the file.


# Main function for Streamlit app
def main():
    st.set_page_config(page_title="img to text")                        # Set the page title for the Streamlit app.

    st.header("Turn image into text story")                             # Display a header in the app.
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")  # Create a file uploader widget for images.

    if uploaded_file is not None:                       # Check if a file has been uploaded.
        print(uploaded_file)                            # Print the uploaded file details.
        bytes_data = uploaded_file.getvalue()           # Read the file data as bytes.
        with open(uploaded_file.name, "wb") as file:    # Open a new file with the uploaded file's name.
            file.write(bytes_data)                      # Write the uploaded file's data to the new file.

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)  # Display the uploaded image in the app.
        scenario = imgtotext(uploaded_file.name)  # Convert the image to text using the imgtotext function.
        chain = Chain()                           # Create an instance of the Chain class.
        story = chain.write_story(scenario)       # Generate a story based on the scenario text.
        texttospeech(story)                       # Convert the story text to speech and save it as an audio file.

        with st.expander("scenario"):   # Create an expandable section to display the scenario text.
            st.write(scenario)          # Display the scenario text.
        with st.expander("story"):      # Create an expandable section to display the story text.
            st.write(story)  

        st.audio("audio.flac")  # Add an audio player to play the generated audio file.

if __name__ == "__main__":  # Check if the script is being run directly.
    main()  # Run the main function.
