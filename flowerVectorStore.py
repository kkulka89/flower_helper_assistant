from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv

class FlowerLoader:

    def __init__(self, csv_path='./Kwiatki.csv'):
        self.csv_loader = CSVLoader(file_path=csv_path, csv_args={
           'delimiter': ',',
           'quotechar': '"',
       })


    def load_flowers(self):
        flowers = self.csv_loader.load()
        # print(len(flowers))
        return flowers

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

class FlowerVectorStore:

    def __init__(self, persist_directory= 'docs/chroma/'):
        load_dotenv()
        self.embedding = OpenAIEmbeddings()
        self.persist_directory  = persist_directory
        # self.vectordb = createFlowerVectorStore() #zrobic singleton

    def create_flower_vector_store(self):
        vectordb = Chroma().from_documents(
            documents=FlowerLoader().load_flowers(),
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )
        # vectordb.persist()
        return vectordb

    def retrieve_flower_vector_store(self):
        vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)

        return vectordb.as_retriever(search_type="similarity")



def main():


    # FlowerEmbeddings().create_flower_vector_store()
    vectordb = FlowerVectorStore().retrieve_flower_vector_store(5)
    print()
    question = "How often i can watering monstera "
    ret = vectordb.invoke(question)
    print(ret)

if __name__ == "__main__":
    main()
