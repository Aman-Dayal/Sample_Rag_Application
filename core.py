
from langchain.chat_models import init_chat_model
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4
from typing_extensions import List, TypedDict
from dotenv import load_dotenv

load_dotenv()

import os

api_key = os.environ['GROQ_API_KEY']
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
config = {"configurable": {"thread_id": "abc123"}}

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class RSYS:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vector_store = InMemoryVectorStore(embedding = self.embeddings)
        self.retriever = self.vector_store.as_retriever()
        self.llm = init_chat_model("llama3-8b-8192", model_provider="groq", api_key = api_key)
        self.chats =[]

    def qa_chain(self):
        system_prompt = """You are a helpful assistant that provides accurate information based ONLY on the provided context. 
            If the information is not in the context, say "I don't have enough information to answer this question."
            Be concise and accurate. Do not make up information or use your general knowledge.
            \nContext:\n
            {context}
            """
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        return question_answer_chain


    def history_chain(self):
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )
        return history_aware_retriever

    def webl(self,urls):

        loader = WebBaseLoader(
        web_paths=(urls),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()
        return docs

    def add_to_vdb(self,urls):
        docs = self.webl(urls)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        chunks = self.vector_store.add_documents(documents=all_splits)
        return len(chunks)
    def main(self,q):
        rag_chain = create_retrieval_chain(self.history_chain(), self.qa_chain())
        ai_msg = rag_chain.invoke({"input": q, "chat_history": self.chats})
        self.chats.extend(
            [
                HumanMessage(content=q),
                AIMessage(content=ai_msg["answer"]),
            ]
        )
        return ai_msg["answer"]