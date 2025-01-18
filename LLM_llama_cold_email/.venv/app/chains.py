import os                                                       # Importing the 'os' module to interact with the operating system (e.g., environment variables)
from langchain_groq import ChatGroq                             # Importing ChatGroq from langchain_groq to use a language model for tasks
from langchain_community.document_loaders import WebBaseLoader  # Importing WebBaseLoader to load web pages 
from langchain_core.prompts import PromptTemplate               # Importing PromptTemplate to define and structure prompts for the language model
from langchain_core.output_parsers import JsonOutputParser      # Importing JsonOutputParser to parse and validate JSON output from the language model
from langchain_core.exceptions import OutputParserException     # Importing OutputParserException to handle exceptions during parsing
from dotenv import load_dotenv                                  # Importing load_dotenv to load environment variables from a .env file

load_dotenv()               # Loading environment variables from the .env file

os.getenv("GROQ_API_KEY")   # Fetching the GROQ_API_KEY environment variable (used as an API key for the ChatGroq model)

class Chain:                # Defining a class named 'Chain' to encapsulate the functionality of the program
    def __init__(self):     # Constructor method to initialize a ChatGroq object with specific parameters
        self.llm = ChatGroq(temperature=0, groq_api_key = os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
        # Setting temperature to 0 for deterministic responses，retrieving API key from environment variables, specifying the model to use

    def extract_jobs(self, cleaned_text):                      # Defining a method to extract job postings from cleaned text
        prompt_extract = PromptTemplate.from_template(         # Creating a prompt template for extracting job postings
            """
            ### SCRAPPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the 
            following keys: `role`, `experience`, `skills`, and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm                           # Combining the prompt with the language model to create a chain
        res = chain_extract.invoke(input={'page_data': cleaned_text})       # Invoking the chain with the provided input text
        try:
            json_parser = JsonOutputParser()                                # Creating a JSON parser to validate and parse the model's output
            res = json_parser.parse(res.content)                            # Parsing the content of the response
        except OutputParserException:                                       # Raising an exception if parsing fails due to context or other issues
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]                      # Ensuring the result is a list (even if a single job is returned)

    def write_mail(self, job):                                              # Defining a method to generate a cold email for a given job description
        prompt_email = PromptTemplate.from_template(                        # Creating a prompt template for generating the email
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Elon, the CEO of multiple big companies like SpaceX, Tesla etc.
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of you
            in fulfilling their needs.
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            """
        )

        chain_email = prompt_email | self.llm                               # Combining the prompt with the language model to create a chain
        res = chain_email.invoke({"job_description": str(job)})             # Invoking the chain with the job description as input
        return res.content                                                  # Returning the content of the generated email

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))                                        # Printing the value of the GROQ_API_KEY environment variable
