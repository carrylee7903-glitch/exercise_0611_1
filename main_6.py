import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# .env 파일 안의 OPENAI_API_KEY 읽기
load_dotenv()

# LangChain 관련 모듈 불러오기
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# [수정 포인트] Chroma 대신 FAISS 불러오기
from langchain_community.vectorstores import FAISS

st.title("PDF File Reader")
st.write("----------------")

uploaded_file = st.file_uploader("PDF 파일을 선택하세요", type=["pdf"])
st.write("----------------")

# [핵심 수정] 매번 DB를 새로 만드는 것을 방지하는 캐싱 적용
@st.cache_resource
def process_pdf_and_create_db(_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, _file.name)
    
    with open(temp_filepath, "wb") as f:
        f.write(_file.getvalue())

    loader = PyPDFLoader(temp_filepath)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings()
    
    # [수정 포인트] Chroma 대신 FAISS로 데이터베이스 생성
    db = FAISS.from_documents(documents=texts, embedding=embeddings)
    
    return db, len(pages), len(texts)

if uploaded_file is not None:
    # 한 번만 실행되어 속도가 훨씬 빠릅니다.
    db, total_pages, total_texts = process_pdf_and_create_db(uploaded_file)
    st.success(f"PDF 로딩 및 DB 구축 완료! (총 {total_pages}페이지 / {total_texts}개 조각)")

    retriever = db.as_retriever(search_kwargs={"k": 3})

    st.header("PDF에게 질문하세요")
    question = st.text_input("질문 입력")

    if st.button("질문하기"):
        if question:
            with st.spinner("답변 생성 중..."):
                # [수정 포인트] 올바른 모델명으로 수정
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0
                )

                prompt = ChatPromptTemplate.from_template(
                    """
                    당신은 PDF 분석 전문가입니다.
                    아래 Context만 이용해서 질문에 답하세요.

                    Context:
                    {context}

                    질문:
                    {input}

                    답변:
                    """
                )

                document_chain = create_stuff_documents_chain(llm, prompt)
                qa_chain = create_retrieval_chain(retriever, document_chain)

                response = qa_chain.invoke({"input": question})

                # 답변 출력
                st.write(response["answer"])
        else:
            st.warning("질문을 입력하세요.")
