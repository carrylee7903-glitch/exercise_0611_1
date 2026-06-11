import os
import tempfile

import streamlit as st
# 추가된 부분: Streamlit 컨텍스트 유지를 위한 모듈
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler

st.title("📄 PDF File Reader")
st.write("----------------")

openai_key = st.text_input("OPENAI_API_KEY", type="password")

uploaded_file = st.file_uploader("PDF 파일을 올려주세요", type=["pdf"])
st.write("----------------")

def pdf_to_document(uploaded_file):
    """   
    Streamlit 업로드 PDF를
    LangChain Document 형태로 변환
    """
    # 임시 폴더 생성
    temp_dir = tempfile.TemporaryDirectory()

    # 임시 PDF 파일
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)

    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader(temp_filepath)
    pages = loader.load()
    return pages

class StreamHandler(BaseCallbackHandler):
    """
    GPT가 토큰을 생성할 때마다
    Streamlit 화면에 출력하는 Handler
    """
    def __init__(self, container):
        self.container = container
        self.text = ""
        # 수정된 부분 1: 핸들러 생성 시점의 컨텍스트를 저장
        self.ctx = get_script_run_ctx()

    def on_llm_new_token(self, token, **kwargs):
        # 수정된 부분 2: 토큰이 생성될 때마다 백그라운드 쓰레드에 컨텍스트 강제 주입
        add_script_run_ctx(ctx=self.ctx)
        
        # 새 토큰 누적
        self.text += token
        # 화면 갱신
        self.container.markdown(self.text)

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    texts = text_splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings(api_key=openai_key)

    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings
    )

    retriever = db.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    st.header("PDF에게 질문하세요")
    question = st.text_input("질문 입력")

    if st.button("질문하기"):
        if question == "":
            st.warning("질문을 입력하세요")
        else:
            with st.spinner("답변 생성중..."): # Streamlit 버전에 따라 show_time 파라미터가 호환 안 될 수 있어 기본값으로 두었습니다. 필요시 복구하세요.
                
                chat_box = st.empty()
                handler = StreamHandler(chat_box)

                # 수정된 부분 3: 모델명 변경 및 콜백 제거
                llm = ChatOpenAI(
                    model="gpt-4o-mini", # gpt-4.1-mini -> gpt-4o-mini 로 변경
                    temperature=0,
                    api_key=openai_key,
                    streaming=True
                )

                prompt = ChatPromptTemplate.from_template(
                    """
                    당신은 PDF 분석 AI 입니다.
                    Context:   {context}
                    Question:  {input}
                    답변:
                    """
                )

                document_chain = create_stuff_documents_chain(llm, prompt)

                qa_chain = create_retrieval_chain(
                    retriever,
                    document_chain
                )

                # 수정된 부분 4: invoke 실행 시 config를 통해 콜백 핸들러 주입
                qa_chain.invoke(
                    {"input": question},
                    config={"callbacks": [handler]}
                )
