import os 
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from utils.config_loader import load_config


class ModelLoader:
    """ 
    A utility class to load embedding models and llm 
    """
    def __init__(self):
        load_dotenv()
        self._validate_env()
        self.config = load_config()
        
    def _validate_env(self):
        """ 
        Validate necessary environment variable
        """
        required_vars = ['GOOGLE_API_KEY', 'GROQ_API_KEY']
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables {missing_vars}")
        
    def load_embeddings(self):
        """ 
        Load and return the embedding model
        """
        print("Loading embedding model")
        model_name = self.config['embedding_model']['model_name']
        return GoogleGenerativeAIEmbeddings(model=model_name)
    
    def load_llm(self):
        """ 
        Load and return embedding model
        """
        print("LLM loading...")
        model_name = self.config['llm']['groq']['model_name']
        groq_model = ChatGroq(groq_api_key=self.groq_api_key, model=model_name)
        
        return groq_model
        