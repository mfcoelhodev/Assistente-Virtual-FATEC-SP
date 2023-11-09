from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatOpenAI
import magic
import os
import nltk
import constants


#cadastro da chave api presente no arquivo constants.py
os.environ["OPENAI_API_KEY"] = constants.APIKEY
openai_api_key = os.environ["OPENAI_API_KEY"]

def Samsung_Chatbot(pergunta):

    # nltk.download('averaged_perceptron_tagger')

    # pip install unstructured
    # Other dependencies to install https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/unstructured_file.html
    # pip install python-magic-bin
    # pip install chromadb
    # carregando diretório 
    loader = DirectoryLoader("data/")
    # carregando textos na variavel documents
    documents = loader.load()
    # preparando função para dividir os textos em partes de 1000 caracteres cada
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # utilizando a função para dividir os textos
    texts = text_splitter.split_documents(documents)
    # transformando partes do texto em embeddings 
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # preparando a procura dos documentos por embeddings através da biblioteca faiss do langchain
    docsearch = FAISS.from_documents(texts, embeddings)
    #Carregando LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    # Criando retriever de informações
    #qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
    # Run a query
    contexto = """
    você irá operar como assistente de suporte da Samsung. Você NÃO DEVE, em hipótese nenhuma, falar de outra empresa a não ser a Samsung.
    Garanta que você possa oferecer suporte ao cliente em relação aos produtos e à empresa de maneira profissional, educada e informativa. 
    Você deve ser estritamente profissional e educado em todas as interações com o cliente. 
    Você não deve falar de outras empresas ou de produtos que não sejam da Samsung. 
    Você não deve falar de outros assuntos triviais não relacionados à empresa. 
    Você não deve inventar fatos, e será honesto caso não tenha acesso à informação pedida.
    É PROIBIDO FALAR DE PRODUTOS DE OUTRAS EMPRESAS.
    """
    qa = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=docsearch.as_retriever(),
                                    return_source_documents=True)
    result = qa({"query": pergunta + "" + contexto})
    resp = result['result']
    #retornando resposta do chatgpt
    return resp

    
