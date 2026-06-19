import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# ---------------------------------------------------------
# CREDENCIAIS (Google AI Studio)
# ---------------------------------------------------------
os.environ["GOOGLE_API_KEY"] = "SUA_CHAVE_GEMINI_AQUI"

# ---------------------------------------------------------
# SETUP BÁSICO (Indexação e Chunking)
# ---------------------------------------------------------
print("1. Carregando e quebrando o documento de texto...")
# Resolve o caminho absoluto do arquivo para evitar erro se rodar de outra pasta
caminho_base = os.path.dirname(os.path.abspath(__file__))
caminho_arquivo = os.path.join(caminho_base, "A_Evolucao_do_RAG.txt")

# Carrega o texto discursivo sobre RAG
loader = TextLoader(caminho_arquivo, encoding="utf-8")
docs = loader.load()

# Quebra o texto em pedaços menores (chunks) para o banco vetorial
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Cria os Embeddings (roda localmente na sua máquina) e o Banco Vetorial na RAM
print("2. Carregando modelo de Embeddings (Isso pode levar alguns segundos)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("3. Criando Banco Vetorial na RAM...")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Instancia a LLM do Google (Gemini) para otimizar as buscas e gerar a resposta final
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0
)

# ---------------------------------------------------------
# PASSO 1: PRÉ-RECUPERAÇÃO (Otimização de Consulta)
# ---------------------------------------------------------
# Define a recuperação inicial (Top K = 15)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

# O MultiQuery usa o Gemini para reescrever a pergunta do usuário de várias 
# formas, garantindo que o banco vetorial encontre a intenção correta.
retriever_pre_otimizado = MultiQueryRetriever.from_llm(
    retriever=base_retriever, 
    llm=llm
)

# ---------------------------------------------------------
# PASSO 2 E 3: PÓS-RECUPERAÇÃO E RE-RANQUEAMENTO
# ---------------------------------------------------------
# Instancia o modelo Cross-Encoder (focado puramente em ranqueamento semântico profundo)
print("4. Carregando modelo Cross-Encoder (Isso também pode demorar um pouquinho)...")
modelo_reranker = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

# Configura o compressor para filtrar os 15 resultados iniciais e manter apenas os 3 melhores (Top N)
compressor = CrossEncoderReranker(model=modelo_reranker, top_n=3)

# Une tudo no pipeline avançado: Consulta -> Pré-recuperação -> Busca FAISS -> Re-ranqueamento
rag_avancado_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=retriever_pre_otimizado
)

# ---------------------------------------------------------
# TESTANDO O PIPELINE
# ---------------------------------------------------------
print("5. Iniciando o processamento da pergunta na LLM (RAG Avançado)...\n")

# Pergunta desafiadora para testar a filtragem do Cross-Encoder
pergunta_usuario = "Por que não devo simplesmente enviar todos os documentos que o banco vetorial encontrou direto para a LLM?"

# O comando invoke dispara todo o fluxo de ponta a ponta
documentos_finais = rag_avancado_retriever.invoke(pergunta_usuario)

print("=== Documentos selecionados e re-ranqueados (Top 3) ===")
for i, doc in enumerate(documentos_finais):
    print(f"\n[ Trecho {i+1} ]")
    print(doc.page_content)
    print("-" * 50)