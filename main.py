import os
import re
import gradio as gr
from dotenv import load_dotenv

from document_loader import DocumentLoader
from vector_store import VectorStoreManager
from agents import AgentManager
from chat_handler import ChatHandler

class ProfileQuerySystem:
    """
    Main application class for Sarwesh's Profile Query System.
    """
    def __init__(self):
        """
        Initialize the Profile Query System.
        """
        # Load environment variables
        load_dotenv()
        os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY')

        # Initialize components
        self.document_loader = DocumentLoader()
        self.vector_store_manager = VectorStoreManager()
        self.agent_manager = AgentManager()
        self.chat_handler = ChatHandler()

        # Load documents
        self.cv_text = self.document_loader.read_pdf("Profile_Query_System/Profile_chatbot/data/CV_Sarwesh_ (1).pdf")
        self.transcript_text = self.document_loader.read_pdf("Profile_Query_System/Profile_chatbot/data/sarwesh_transcript.pdf")
        self.publication_text = self.document_loader.read_pdf("Profile_Query_System/Profile_chatbot/data/causality between sentiment and crypto currency prices.pdf")
        self.mitacs_text = self.document_loader.read_pdf("Profile_Query_System/Profile_chatbot/data/MITACS_RESEARCH.pdf")
        self.snowflake_text = self.document_loader.read_docx("Profile_Query_System/Profile_chatbot/data/Query Intent.docx")+"\n\n"
        self.snowflake_text +=self.document_loader.read_docx("Profile_Query_System/Profile_chatbot/data/Document Diversity.docx")
        

        # Create vector stores
        self.publication_vector_store = self.vector_store_manager.create_vector_store(
            self.publication_text
        )
        self.snowflake_vector_store = self.vector_store_manager.create_vector_store(
            self.snowflake_text
        )

        # Initialize agents
        self.answerability_agent = self.agent_manager.create_answerability_agent()
        self.data_identifier_agent = self.agent_manager.create_data_identifier_agent()
        self.main_agent = self.agent_manager.create_main_agent()
        self.transcript_agent = self.agent_manager.create_transcript_agent()
        self.publication_agent = self.agent_manager.create_publication_agent()
        self.mitacs_agent = self.agent_manager.create_mitacs_agent()
        self.snowflake_agent = self.agent_manager.create_snowflake_agent()

    def route_query(self, input_dict, session_id):
        """
        Route the query to the appropriate agent based on context and data source.
        
        Args:
            input_dict (dict): Input dictionary containing query and context
            session_id (str): Current session identifier
        
        Returns:
            str: Response from the appropriate agent
        """
        # Get chat history for the session
        chat_history = self.chat_handler.get_session_history(session_id)
        
        # Prepare input for answerability check
        agent_input = {
            "input": input_dict["input"],
            "cv": self.cv_text,
            "chat_history": chat_history.messages
        }
        
        
        # Check if the question can be answered using the CV
        answerability_response = self.answerability_agent.invoke(agent_input)
        
        if answerability_response.strip().lower() == "yes":
            # Use main agent to answer from CV
            response = self.main_agent.invoke(input_dict)
        else:
            # Identify which additional data source is needed
            data_identifier_input = {
                "input": input_dict["input"],
                "cv": self.cv_text
            }
            data_identifier_response = self.data_identifier_agent.invoke(data_identifier_input)
            
            # Extract the number from the response
            match = re.search(r'\b[1-6]\b', data_identifier_response)
            
            if match:
                number = int(match.group())
                
                # Route to appropriate agent based on the number
                if number == 1:
                    # Transcript agent
                    transcript_input = {
                        "input": input_dict["input"],
                        "transcript": self.transcript_text
                    }
                    response = self.transcript_agent.invoke(transcript_input)
                
                elif number == 2:
                    # Publication agent
                    publication_context = self.vector_store_manager.retrieve_relevant_chunks(
                        input_dict["input"], 
                        self.publication_vector_store
                    )
                    publication_input = {
                        "input": input_dict["input"],
                        "publication_context": publication_context
                    }
                    response = self.publication_agent.invoke(publication_input)
                
                elif number == 3:
                    # Snowflake agent
                    snowflake_context = self.vector_store_manager.retrieve_relevant_chunks(
                        input_dict["input"], 
                        self.snowflake_vector_store
                    )
                    snowflake_input = {
                        "input": input_dict["input"],
                        "snowflake_context": snowflake_context
                    }
                    response = self.snowflake_agent.invoke(snowflake_input)
                
                elif number == 4:
                    # Mitacs agent
                    mitacs_input = {
                        "input": input_dict["input"],
                        "mitacs_text": self.mitacs_text
                    }
                    response = self.mitacs_agent.invoke(mitacs_input)
                
                elif number == 5:
                    response = "I don't have the relevant information for the query."
                
                elif number == 6:
                    response = "Please ask questions related to Sarwesh."
                
                else:
                    response = "Error in routing the query."
            else:
                response = "Unable to determine the appropriate data source."
        
        # Add messages to chat history
        chat_history.add_user_message(input_dict["input"])
        chat_history.add_ai_message(response)
        
        return response

    def chat_interaction(self, message, history):
        """
        Handle chat interactions for the profile chatbot.
        
        Args:
            message (str): User's input message
            history (list): Chat history of previous interactions
        
        Returns:
            str: Bot's response
        """
        # Generate a unique session ID
        session_id = self.chat_handler.generate_session_id()
        
        try:
            # Route the query
            response = self.route_query(
                {
                    "input": message
                },
                session_id
            )
            
            return response
        
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def launch_chat_interface(self):
        """
        Launch the Gradio chat interface.
        """
        iface = gr.ChatInterface(
            fn=self.chat_interaction,
            title="Sarwesh Profile Chatbot",
            description=(
                "Ask questions about Sarwesh's profile, experience, education, "
                "and more. The chatbot will use his resume and additional "
                "contextual documents to provide comprehensive answers."
            ),
            theme="soft",
            examples=[
                "Where did Sarwesh do his college?",
                "Tell me about his work experience",
                "What projects has he worked on?",
                "Can you share details about his publication?"
            ],
            cache_examples=False
        )
        
        iface.launch(share=True, debug=True)

def main():
    """
    Main function to set up and launch the chatbot interface.
    """
    profile_query_system = ProfileQuerySystem()
    profile_query_system.launch_chat_interface()

if __name__ == "__main__":
    main()