import uuid
from langchain_core.chat_message_histories import ChatMessageHistory

class ChatHandler:
    """
    Manages chat sessions and interactions.
    """
    def __init__(self):
        """
        Initialize the chat handler with a session store.
        """
        self.store = {}

    def generate_session_id(self):
        """
        Generate a unique session ID for each conversation.
        
        Returns:
            str: Unique session identifier.
        """
        return str(uuid.uuid4())

    def get_session_history(self, session_id):
        """
        Retrieve or create a chat session history.
        
        Args:
            session_id (str): Session identifier.
        
        Returns:
            ChatMessageHistory: Session's message history.
        """
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def reset_session_history(self, session_id):
        """
        Reset the chat history for a specific session.
        
        Args:
            session_id (str): Session identifier to reset.
        """
        self.store[session_id] = ChatMessageHistory()
        print(f"Chat history for session '{session_id}' has been reset.")