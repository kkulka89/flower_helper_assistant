from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

from flowerVectorStore import FlowerVectorStore

class ChatBot():

    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.chain = self.create_conversional_chain()
        self.store = {}

    #First part of chain only related to retrieve from vectorbase
    #can be replacec by create_csv_agent  langchain_experimental.agents.agent_toolkits import create_csv_agent (based on pandas)/verify
    #with bigger database sql query from db can be better
    def create_rag_chain(self):
        retriever = FlowerVectorStore().retrieve_flower_vector_store()

        contextualize_q_system_prompt = """Given a chat history and the latest user question \
                which might reference context in the chat history, formulate a standalone question related to flowers \
                which can be understood without the chat history. Do NOT answer the question, \
                just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )
        return history_aware_retriever

    #main part of chatbot
    def create_question_answer_chain(self):
        qa_system_prompt = """
            You are AI assistant Flowerbot2000 specialized in Flowers.
            Your main task is find the best suitable flower based on context.
            Your secondary task is providing information about proposed flower.
            Collects the necessary information:
             - budget,
             - available space for flower,
             - how often can watering flower

            CONTEXT: 
            {context}
            
            Respond in a short, very conversational friendly style. 
            
            """

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        model =  ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

        return question_answer_chain


    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
            print(self.store[session_id])
        return self.store[session_id]


    def create_conversional_chain(self):
        rag_chain = create_retrieval_chain(self.create_rag_chain(), self.create_question_answer_chain())
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )


        return conversational_rag_chain

    def invoke (self, query):
        return self.chain.invoke({"input": query}, config={
            "configurable": {"session_id": "abc1235"}
        })['answer']

if __name__ == "__main__":
    chat = ChatBot()
    while True:
        query = input()
        if query.lower() == "exit":
            break
        res = chat.invoke(query)
        print(F"\nResponse: {res}")

