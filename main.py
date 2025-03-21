from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage

from flowerVectorStore import FlowerVectorStore
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

load_dotenv()
def chat1(input_str):
    # Define memory for chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Define chatbot prompt template
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "user_input"],
        template="""
        You are a helpful gardening assistant that recommends flowers based on user preferences.
        Consider the user's watering habits, available space, and budget.
        Chat history:
        {chat_history}

        User: {user_input}
        Assistant:
        """
    )

    # Create RAG-based chatbot
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=FlowerVectorStore().retrieve_flower_vector_store(5),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
    )
    response = rag_chain.invoke({"user_input": input_str})
    print("Bot:", response["answer"])



def main0():
    load_dotenv()
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=FlowerVectorStore().retrieve_flower_vector_store()
    )
    question = "Which flower should be watered twice a week?"
    result = qa_chain({"query": question})

    res = result["result"]

    print(res)


def main1(input_str,chat_history):

    llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")

    # qa_chain = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever=FlowerVectorStore().retrieve_flower_vector_store()
    # )


    # Build prompt

    flower_bot_template = """
    You are AI assistant Flowerbot2000 specialized in Flowers  
    Your main task is find the best suitable flower based on context and chat_history
    
    Collects the necessary information:
     - budget,
     - available space for flower,
     - how often can watering flower
   
    CONTEXT: {context}
           

    Respond in a short, very conversational friendly style. 
    """

    # qa_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system":flower_bot_template),
    #         (MessagesPlaceholder("chat_history"),
    #         ("human", "{question}"]),
    #       )
    qa_pormpt = ChatPromptTemplate.from_messages([

        ("system",flower_bot_template),
        MessagesPlaceholder("chat_history"),
        # ("ai", "Hello! I'm Flowerbot2000, your friendly flower assistant. How can I assist you today? "),
        # ("human", "I am looking for a plant for me"),
        ("human", "{question}"),
        ]
    )

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question", "chat_history"],template=flower_bot_template)

    # memory = ConversationBufferMemory(
    # memory_key="chat_history",
    # return_messages=True,
    # )
    # Run chain

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=FlowerVectorStore().retrieve_flower_vector_store(5),
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": qa_pormpt},
    )

    result = qa_chain.invoke({"question": input_str, "chat_history":chat_history})
    answer = result['answer']
    chat_history.extend(F"human:{input_str}\n ai:{answer}")

    print(f" chat hist - {chat_history}")
    print(answer)
    print()
if __name__ == "__main__":
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chat_history = {}
    while True:
        query = input()
        if query.lower() == "exit":
            break
        main1(query, chat_history)
        print("\nResponse:")

