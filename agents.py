import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_message_histories import ChatMessageHistory
#from chat_handler import ChatHandler

class AgentManager:
    """
    Manages different specialized agents for query processing and routing.
    """
    def __init__(self, model="gemini-1.5-flash"):
        """
        Initialize agents with a specific language model.
        
        Args:
            model (str): Language model to use.
        """
        self.llm = ChatGoogleGenerativeAI(model=model)
        
    
    def create_answerability_agent(self):
        """
        Create an agent to determine if a question can be answered 
        using the provided CV.
        
        Returns:
            Runnable: Answerability checker agent chain.
        """
        system_prompt = (
            "You are an assistant that determines whether a given question can be answered using the provided CV and chat history."
            "Respond with a single word: 'yes' or 'no'."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Can the following question be answered using the provided CV and chat history?"
             "Answer with 'yes' or 'no'.\n\nCV: {cv}\n\nChat History: {chat_history}\n\nQuestion: {input}")
        ])
        
        return (
            RunnablePassthrough.assign()
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def create_data_identifier_agent(self):
        """
        Create an agent to identify which data source is relevant.
        
        Args:
            cv_text (str): CV text for context.
        
        Returns:
            Runnable: Data identifier agent chain.
        """
        system_prompt = (
            "You are an assistant that determines which additional data source is relevant:\n"
            "1. Sarwesh's Transcript: Contains all the College coursework and academic details\n"
            "2. Publication Report: Contains the published research paper\n"
            "3. Snowflake Report: Contains the report of the Snowflake project\n"
            "4. Mitacs Report: Contains the report of the Mitacs project\n"
            "5. None of the above four data are relevant to the question\n"
            "6. The Question is not relevant to Sarwesh\n\n"
            "Respond with the number corresponding to the required data source.\n"
            "Here is Sarwesh's Resume for your reference: {cv}\n"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Given the following question, identify which additional data source is needed: {input}")
        ])
        
        return (
            RunnablePassthrough.assign()
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def create_transcript_agent(self):
        """
        Create an agent to analyze Sarwesh's academic transcript.
        
        Returns:
            Runnable: Transcript analysis agent chain.
        """
        system_prompt = (
            "You are an assistant that analyses Sarwesh's Academic Transcript and answers the  questions related to it. \n\n"
            "Here is Sarwesh's Academic Transcript : \n {transcript}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Analyze the transcript and answer the following question.\n\nQuestion: {input}\n\n")
        ])
        
        return (
            RunnablePassthrough.assign()
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def create_publication_agent(self):
        """
        Create an agent to handle queries about publications.
        
        Returns:
            Runnable: Publication agent chain.
        """
        system_prompt = (
            "You are a helpful assistant that helps in answering any questions regarding to the publication of Sarwesh."
            "Use Sarwesh's resume and the following publication report for answering any question."
            "Here is the resume {cv}\n\nPublication Report Context:\n{publication_context}\n\n"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Question: {input}")
        ])
        
        return (
            RunnablePassthrough.assign()
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def create_mitacs_agent(self):
        """
        Create an agent to handle queries about the Mitacs project.
        
        Returns:
            Runnable: Mitacs project agent chain.
        """
        system_prompt = (
            "You are a helpful assistant that helps in answering any questions regarding to the Mitacs Intern/ project of Sarwesh."
            "Use Sarwesh's resume and the following mitacs research report for answering any question."
            "Here is the resume {cv}\n\n mitacs research Report :\n{mitacs_text}\n\n"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Question: {input}")
        ])
        
        return (
            RunnablePassthrough.assign()
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def create_snowflake_agent(self):
        """
        Create an agent to handle queries about the Snowflake project.
        
        Returns:
            Runnable: Snowflake project agent chain.
        """
        system_prompt = (
            "You are a helpful assistant that helps in answering any questions regarding to the Snowflake intern of Sarwesh."
            "Use Sarwesh's resume and the following snowflake work report for answering any question."
            "Here is the resume {cv}\n\n Snowflake intern Report Context:\n{snowflake_context}\n\n"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Question: {input}")
        ])
        
        return (
            RunnablePassthrough.assign()
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def create_main_agent(self):
        """
        Create the main conversational agent.
        
        Returns:
            Runnable: Main conversational agent chain.
        """
        system_prompt = (
            "You are a helpful assistant that helps in answering questions "
            "about Sarwesh's profile, experience, and background. "
            "Provide comprehensive and accurate responses."
            "Use Sarwesh's resume for answering any question. Here is the resume {cv}\n\n Chat History :{chat_history}\n"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        # Base chain without history tracking
        base_chain = (
            RunnablePassthrough.assign(
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )
        

        
        return base_chain