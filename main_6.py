import os
import tempfile

import streamlit as st
# 추가됨: 에러 1 (NoSessionContext) 해결을 위한 모듈
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
    temp_dir = tempfile.TemporaryDirectory()
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
        # 에러 1 해결: 정상 상태일 때의 화면 주소(Context) 기억하기
        self.ctx = get_script_run_ctx()

    def on_llm_new_token(self, token, **kwargs):
        # 에러 1 해결: 백그라운드에서 글자를 쓸 때 주소(Context) 쥐어주기
        add_script_run_ctx(ctx=self.ctx)
        
        self.text += token
        self.container.markdown(self.text)


if uploaded_file is not None:
    # 에러 3 (API 키 누락) 해결: 키가 없으면 여기서 멈춤 (빨간 에러 방지)
    if not openai_key:
        st.info("👈 진행하려면 먼저 상단에 OPENAI_API_KEY를 입력해 주세요.")
        st.stop()

    pages = pdf_to_document(uploaded_file)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    texts = text_splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings(api_key=openai_key)

    # 에러 2 (SyntaxError 괄호 빠짐) 해결 확인 완료
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
            with st.spinner("답변 생성중..."): 

                chat_box = st.empty()
                handler = StreamHandler(chat_box)

                # 에러 4 (모델명 오타) 수정: gpt-4.1-mini -> gpt-4o-mini
                llm = ChatOpenAI(
                    model="gpt-4o-mini", 
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

                # 콜백 핸들러를 실행(invoke) 시점에 안전하게 주입
                qa_chain.invoke(
                    {"input": question},
                    config={"callbacks": [handler]}
                )
