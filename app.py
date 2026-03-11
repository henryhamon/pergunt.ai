import os
import sys
import subprocess

def instalar_dependencias():
    pacotes = ["streamlit", "chromadb", "requests"]
    for pacote in pacotes:
        try:
            __import__(pacote)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pacote, "-q"])

instalar_dependencias()

import streamlit as st
import chromadb
import requests

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "aya-expanse:8b"

st.set_page_config(page_title="RAG com Ollama e ChromaDB", layout="centered")

@st.cache_resource
def carregar_chromadb():
    cliente = chromadb.PersistentClient(path="./chroma_db")
    colecao = cliente.get_or_create_collection(name="documentos_rag")
    return colecao

colecao = carregar_chromadb()

def gerar_embedding(texto):
    resposta = requests.post(OLLAMA_EMBED_URL, json={
        "model": MODEL_NAME,
        "prompt": texto
    })
    if resposta.status_code == 200:
        return resposta.json().get("embedding")
    return None

def gerar_resposta_ollama(prompt):
    resposta = requests.post(OLLAMA_GENERATE_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })
    if resposta.status_code == 200:
        return resposta.json().get("response")
    return "Erro ao gerar resposta no Ollama."

st.sidebar.title("Navegação")
pagina = st.sidebar.radio("Selecione a página:", ["Adicionar Embeddings", "Consulta RAG"])

if pagina == "Adicionar Embeddings":
    st.title("Adicionar ao Banco de Conhecimento")
    st.write("Insira um texto para ser processado e salvo no ChromaDB.")
    
    texto_input = st.text_area("Texto:", height=200)
    
    if st.button("Salvar Texto"):
        if texto_input.strip():
            with st.spinner("Gerando embeddings e salvando..."):
                embedding = gerar_embedding(texto_input)
                if embedding:
                    doc_id = f"doc_{colecao.count() + 1}"
                    colecao.add(
                        embeddings=[embedding],
                        documents=[texto_input],
                        ids=[doc_id]
                    )
                    st.success("Texto salvo com sucesso no banco de embeddings!")
                else:
                    st.error("Erro: falha ao gerar o embedding no Ollama.")
        else:
            st.warning("Por favor, digite algum texto antes de salvar.")

elif pagina == "Consulta RAG":
    st.title("Consulta RAG")
    st.write("Faça uma pergunta. O sistema buscará no banco de embeddings e gerará a resposta com Ollama.")
    
    pergunta = st.text_input("Sua pergunta:")
    
    if st.button("Enviar"):
        if pergunta.strip():
            with st.spinner("Buscando contexto e gerando resposta..."):
                query_embed = gerar_embedding(pergunta)
                
                if query_embed:
                    resultados = colecao.query(
                        query_embeddings=[query_embed],
                        n_results=3
                    )
                    
                    documentos_recuperados = resultados.get("documents", [[]])[0]
                    contexto = "\n\n".join(documentos_recuperados)
                    
                    if not contexto.strip():
                        contexto = "Nenhum contexto encontrado no banco de dados para esta pergunta."
                    
                    prompt = f"""Responda à pergunta do usuário usando exclusivamente as informações presentes no contexto fornecido.

Se o contexto não contiver dados diretamente relacionados à pergunta, responda:
'Ainda não tenho conhecimento sobre esse assunto na base de informações.'
A resposta deve ser curta, clara e humanizada, sem parecer gerada por uma IA.
Use português brasileiro e mantenha tom neutro.
Quando houver informação, reformule-a de forma natural e, se possível, cite trechos relevantes do contexto.
Não faça deduções ou inferências além do que está explicitamente no contexto.

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta:"""
                    
                    resposta_final = gerar_resposta_ollama(prompt)
                    
                    st.subheader("Resposta Gerada")
                    st.write(resposta_final)
                    
                    with st.expander("Ver contexto recuperado do ChromaDB"):
                        for i, doc in enumerate(documentos_recuperados, 1):
                            st.markdown(f"**Trecho {i}:**\n{doc}")
                else:
                    st.error("Erro: falha ao gerar o embedding da pergunta.")
        else:
            st.warning("Por favor, faça uma pergunta.")
