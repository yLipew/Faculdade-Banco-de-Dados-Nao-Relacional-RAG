# 🤖 Atividade Prática — RAG Avançado com LangChain + Gemini + FAISS

> **Disciplina:** Banco de Dados NoSQL  
> **Curso:** Análise e Desenvolvimento de Sistemas — FATESG  
> **Semestre:** 2º Semestre · 2026/1  

![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-000000?style=for-the-badge&logo=langchain&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-VectorStore-0080FF?style=for-the-badge&logo=meta&logoColor=white)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen?style=for-the-badge)

---

## 📌 Sobre o Projeto

Este projeto implementa um pipeline de **RAG Avançado (Retrieval-Augmented Generation)** em Python, evoluindo do modelo "Naive RAG" para uma arquitetura com **pré-recuperação** e **pós-recuperação** otimizadas. O documento base é um texto discursivo sobre a própria evolução do RAG, servindo como prova de conceito do conceito que ele mesmo descreve.

---

## 🧱 Arquitetura do Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG AVANÇADO PIPELINE                        │
├───────────────┬─────────────────────────────┬───────────────────────┤
│  INDEXAÇÃO    │     PRÉ-RECUPERAÇÃO          │   PÓS-RECUPERAÇÃO     │
│               │                             │                       │
│  TextLoader   │  MultiQueryRetriever        │  CrossEncoderReranker │
│      ↓        │  (Gemini reescreve a query) │  (Top K=15 → Top N=3) │
│  Chunking     │          ↓                  │          ↓            │
│  (500/50)     │     FAISS Search            │   Contexto Final      │
│      ↓        │     (Top K = 15)            │       → LLM           │
│  HuggingFace  │                             │                       │
│  Embeddings   │                             │                       │
│      ↓        │                             │                       │
│  FAISS Index  │                             │                       │
└───────────────┴─────────────────────────────┴───────────────────────┘
```

---

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Versão / Modelo | Função |
|---|---|---|
| Python | 3.13+ | Linguagem principal |
| LangChain | latest | Orquestração do pipeline RAG |
| Google Gemini | gemini-2.5-flash | LLM para reescrita de query e resposta final |
| HuggingFace | all-MiniLM-L6-v2 | Geração de embeddings (local) |
| FAISS | faiss-cpu | Banco vetorial em memória RAM |
| Cross-Encoder | ms-marco-MiniLM-L-6-v2 | Re-ranqueamento semântico profundo |

---

## 📁 Estrutura do Projeto

```
RAG-avancado_Cassandra-API/
└── Atividade_RAG-Avancado/
    ├── Advanced_RAG.py          # Pipeline principal RAG Avançado
    ├── A_Evolucao_do_RAG.txt    # Documento base para indexação
    ├── requirements.txt         # Dependências do projeto
    ├── output.log               # Log de execução
    ├── output2.log              # Log de execução (2ª rodada)
    ├── pip_freeze.txt           # Snapshot do ambiente
    ├── RAG-1.png                # Print de evidência
    ├── RAG-2.png                # Print de evidência
    └── README.md
```

---

## ⚙️ Configuração do Ambiente

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/RAG-avancado_Cassandra-API.git
cd RAG-avancado_Cassandra-API/Atividade_RAG-Avancado
```

### 2. Crie e ative o ambiente virtual

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
langchain
langchain-community
langchain-text-splitters
langchain-google-genai
faiss-cpu
sentence-transformers
```

### 4. Configure a chave do Google AI Studio

```python
# No arquivo Advanced_RAG.py, substitua:
os.environ["GOOGLE_API_KEY"] = "SUA_CHAVE_GEMINI_AQUI"
```

Ou via variável de ambiente (recomendado):

```bash
# Linux/macOS
export GOOGLE_API_KEY="sua_chave_aqui"

# Windows (PowerShell)
$env:GOOGLE_API_KEY="sua_chave_aqui"
```

> ⚠️ Nunca versione sua `GOOGLE_API_KEY` diretamente no código. Use variáveis de ambiente ou um `.env` no `.gitignore`.

---

## ▶️ Execução

```bash
python Advanced_RAG.py
```

### Saída esperada no terminal

```
1. Carregando e quebrando o documento de texto...
2. Carregando modelo de Embeddings (Isso pode levar alguns segundos)...
Loading weights: 100% ████████████ 103/103 [2224.06it/s]
3. Criando Banco Vetorial na RAM...
4. Carregando modelo Cross-Encoder (Isso também pode demorar um pouquinho)...
Loading weights: 100% ████████████ 105/105 [2390.77it/s]
5. Iniciando o processamento da pergunta na LLM (RAG Avançado)...

=== Documentos selecionados e re-ranqueados (Top 3) ===

[ Trecho 1 ]
A técnica mais poderosa da pós-recuperação é o Re-ranqueamento...
--------------------------------------------------

[ Trecho 2 ]
Já a fase de pós-recuperação lida com o refinamento dos resultados...
--------------------------------------------------

[ Trecho 3 ]
A Geração Aumentada por Recuperação, ou RAG (Retrieval-Augmented Generation)...
--------------------------------------------------
```

---

## 🔍 Como Funciona — Passo a Passo

### Etapa 1 — Indexação

```python
# Carrega o .txt e divide em chunks de 500 chars com overlap de 50
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Gera embeddings localmente (sem custo de API)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
```

### Etapa 2 — Pré-recuperação (MultiQuery)

```python
# Gemini reescreve a pergunta de múltiplas formas para ampliar o recall
retriever_pre_otimizado = MultiQueryRetriever.from_llm(
    retriever=base_retriever,  # Top K = 15
    llm=llm
)
```

### Etapa 3 — Pós-recuperação (Re-ranqueamento)

```python
# Cross-Encoder avalia semanticamente cada chunk e filtra os Top 3
compressor = CrossEncoderReranker(model=modelo_reranker, top_n=3)
rag_avancado_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever_pre_otimizado
)
```

---

## 🆚 RAG Naive vs RAG Avançado

| Característica | RAG Naive | RAG Avançado |
|---|---|---|
| Otimização de query | ❌ Nenhuma | ✅ MultiQueryRetriever |
| Busca vetorial | Similaridade de cosseno | Similaridade de cosseno |
| Filtragem pós-busca | ❌ Envia todos os chunks | ✅ Cross-Encoder Reranker |
| Chunks para a LLM | Top K (todos) | Top N=3 (os melhores) |
| Custo computacional | Baixo | Médio (modelos locais) |
| Qualidade da resposta | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🧠 Conceitos Aprendidos

- ✅ Diferença entre RAG Naive e RAG Avançado
- ✅ Chunking com overlap para manter contexto entre trechos
- ✅ Embeddings locais com HuggingFace (sem custo de API)
- ✅ Banco vetorial in-memory com FAISS
- ✅ Query Expansion via `MultiQueryRetriever` com Gemini
- ✅ Re-ranqueamento semântico profundo com `CrossEncoderReranker`
- ✅ Orquestração de pipeline com LangChain (`ContextualCompressionRetriever`)

---

## 📝 .gitignore recomendado

```
.venv/
__pycache__/
*.pyc
.env
*.env
output*.log
pip_freeze.txt
```

---

## 📄 Licença

Projeto acadêmico desenvolvido para fins educacionais — FATESG · 2026.
