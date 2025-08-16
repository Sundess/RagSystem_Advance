import os
import google.generativeai as genai


class GeminiChat:
    """
    Manages Google Gemini AI for generating answers
    """

    def __init__(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def generate_answer_with_context(self, query, context):
        """
        Generate an answer using Gemini with provided context
        """
        prompt = f"""
        You are a helpful AI assistant. Answer the user's question based on the provided context.
        
        Context Information:
        {context}
        
        User Question: {query}
        
        Instructions:
        - Answer the question using only the information provided in the context
        - If the context doesn't contain enough information to answer the question, say so clearly
        - Be concise and accurate
        - If you need to make assumptions, state them clearly
        - Provide specific examples from the context when relevant
        
        Answer:
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

    def generate_simple_answer(self, query):
        """
        Generate a simple answer without specific context
        """
        try:
            response = self.model.generate_content(query)
            return response.text
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

    def chat_with_history(self, messages):
        """
        Chat with conversation history
        """
        try:
            chat = self.model.start_chat(history=messages[:-1])
            response = chat.send_message(messages[-1]['parts'][0])
            return response.text
        except Exception as e:
            return f"❌ Error in chat: {str(e)}"
