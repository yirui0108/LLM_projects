import streamlit as st                                             # Importing Streamlit for building web apps with a simple Python interface
from langchain_community.document_loaders import WebBaseLoader     # Importing WebBaseLoader to load content from a URL


from chains import Chain               # Importing the Chain class from the 'chains' module
from utils import clean_text           # Importing the clean_text function from the 'utils' module


def create_streamlit_app(llm, clean_text):     # Defining a function to create the Streamlit app, taking 'llm' and 'clean_text' as inputs
    st.title("📧 Cold Mail Generator")         # Setting the title of the Streamlit app with an emoji for design
    # Creating a text input field for the user to input a URL with a default value
    url_input = st.text_input("Enter a URL:", value="https://seagatecareers.com/job/Shakopee-Biochemistry-Intern-MN/1224963100/")
    submit_button = st.button("Submit")         # Adding a "Submit" button to the app

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])                  # Creating an instance of WebBaseLoader with the input URL
            data = clean_text(loader.load().pop().page_content)  # Loading the webpage content, cleaning it with the clean_text function, and extracting the text
            jobs = llm.extract_jobs(data)                        # Using the language model to extract job information from the cleaned text
            for job in jobs:
                email = llm.write_mail(job,)                     # Generating a cold email for the job using the language model
                st.code(email, language='markdown')              # Displaying the generated email in the app using Markdown formatting
        except Exception as e:                                   # Handling exceptions that may occur during processing
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    chain = Chain()                                                                        # Creating an instance of the Chain class to use its functionalities
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")   # Setting the configuration for the Streamlit app: wide layout, a title, and an icon
    create_streamlit_app(chain, clean_text)                                                # Calling the create_streamlit_app function to build and run the app
