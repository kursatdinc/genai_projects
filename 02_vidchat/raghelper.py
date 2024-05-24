from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

my_key_openai = os.getenv("openai_apikey")
my_key_google = os.getenv("google_apikey")
my_key_hf = os.getenv("huggingface_access_token")

llm_gemini = ChatGoogleGenerativeAI(google_api_key=my_key_google, model="gemini-pro")
llm_openai = ChatOpenAI(api_key=my_key_openai, model="gpt-4-1106-preview")

embeddings = OpenAIEmbeddings(api_key=my_key_openai)

# embeddings = HuggingFaceInferenceAPIEmbeddings(
#     api_key=my_key_hf,
#     model_name="sentence-transformers/all-MiniLM-l6-v2"
# )

#1 Dil modeliyle konuşma
def ask_gemini(prompt):

    AI_Response = llm_gemini.invoke(prompt)

    return AI_Response.content

def ask_gpt(prompt):

    AI_Response = llm_openai.invoke(prompt)

    return AI_Response.content


#2 Bellek genişletme süreci - video transkript metinleri

def rag_with_video_transcript(transcript_docs, prompt):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len
    )

    splitted_documents = text_splitter.split_documents(transcript_docs)

    vectorstore = FAISS.from_documents(splitted_documents, embeddings)
    retriever = vectorstore.as_retriever()

    relevant_documents = retriever.get_relevant_documents(prompt)

    context_data = ""

    for document in relevant_documents:
        context_data = context_data + " " + document.page_content

    final_prompt = f"""Şöyle bir sorum var: {prompt}
    Bu soruyu yanıtlamak için elimizde şu bilgiler var: {context_data} .
    Bu sorunun yanıtını vermek için yalnızca sana burada verdiğim eldeki bilgileri kullan. Bunların dışına asla çıkma.
    """

    AI_Response = ask_gpt(final_prompt)

    return AI_Response, relevant_documents