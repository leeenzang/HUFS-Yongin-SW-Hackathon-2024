
import os
import fitz  # PyMuPDF, PDF 텍스트 추출용
import streamlit as st  # Streamlit을 사용하여 웹 인터페이스 생성
import time  # 타이핑 효과를 위한 딜레이 설정

# LangChain 및 OpenAI API 관련 import
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory, AIMessage, HumanMessage
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import RetrievalQA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.cache import InMemoryCache  # LangChain 메모리 캐시 추가

os.environ['OPENAI_API_KEY'] = 'API_Key'
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ["LANGCHAIN_API_KEY"] = 'API_Key'
os.environ["LANGCHAIN_PROJECT"] = 'aiproject1'

# PDF 파일의 각 페이지에서 텍스트를 추출하여 리스트로 반환하는 함수
def extract_text_by_pages(pdf_path: str):
    pdf_document = fitz.open(pdf_path)
    pages_text = []
    
    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)
        pages_text.append(page.get_text("text"))
    
    pdf_document.close()
    return pages_text

# LangChain 모델 설정
llm = ChatOpenAI(model="gpt-4o-mini")

# Streamlit에서 사용자 인터페이스 생성
st.title("용인시 스마트팜 챗봇🌱")
st.write("용인시 스마트팜 창업인을 위한 가이드 챗봇입니다.🧑🏻‍🌾 궁금한 점을 입력하세요!")

# PDF 경로 리스트
yongin_pdfs = ["news.pdf", "용인병해충.pdf", "용인지원.pdf"]
smartfarm_pdfs = ["smartfarm_guide_1_OCR.pdf", "smartfarm_guide_2_OCR.pdf"]

# PDF 텍스트를 추출하여 초기화 (캐시 적용)
pdfs_text_by_page = {pdf_path: extract_text_by_pages(pdf_path) for pdf_path in (yongin_pdfs + smartfarm_pdfs)}
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

# 이전 대화 히스토리 출력
for chat in st.session_state["conversation_history"]:
    role, content = chat.split(":", 1)
    if role == "User":
        st.chat_message("user").write(content.strip())
    elif role == "AI":
        st.chat_message("assistant").write(content.strip())

# 질문 입력 및 대화 표시
if prompt := st.chat_input("질문을 입력하세요:"):
    # 사용자 입력을 히스토리에 추가하고 화면에 표시
    st.session_state["conversation_history"].append(f"User: {prompt}")
    with st.chat_message("user"):
        st.write(prompt)
    
    # '용인' 키워드가 포함된 질문인지 확인하여 탐색할 PDF 파일 설정
    selected_pdfs = yongin_pdfs if "용인" in prompt else smartfarm_pdfs

    # 유사도 기반 페이지 선택
    max_similarity = 0
    best_page = None
    best_pdf_path = None

    # 선택한 PDF 파일에서만 유사도 계산
    for pdf_path in selected_pdfs:
        pages_text = pdfs_text_by_page[pdf_path]
        vectorizer = TfidfVectorizer().fit_transform([prompt] + pages_text)
        vectors = vectorizer.toarray()
        
        # query와 페이지 텍스트 간의 유사도 계산
        cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        best_index = cosine_similarities.argmax()
        similarity = cosine_similarities[best_index]

        if similarity > max_similarity:
            max_similarity = similarity
            best_page = pages_text[best_index]
            best_pdf_path = pdf_path

    if best_page:
        # 역할 지침과 히스토리를 포함하여 프롬프트 생성
        role_prompt = (
            "You are a specialized chatbot in smart farming and startup support for Yongin City. "
            "Answer questions based on provided materials, focusing on smart farm entrepreneurship. "
        )
        conversation_text = role_prompt + "\n".join(st.session_state["conversation_history"]) + f"\n\nMaterial:\n{best_page}"
        
        message = HumanMessage(content=conversation_text)
        response = llm.invoke([message])
        
        # AI 응답을 히스토리에 추가
        st.session_state["conversation_history"].append(f"AI: {response.content}")

        # 한 글자씩 타이핑 효과로 AI 응답 출력
        with st.chat_message("assistant"):
            typing_text = ""
            placeholder = st.empty()
            for char in response.content:
                typing_text += char
                placeholder.write(typing_text)
                time.sleep(0.05)  # 타이핑 속도 조정
    else:
        st.session_state["conversation_history"].append("AI: 관련된 정보를 찾을 수 없습니다. 다시 시도해주세요.")
        with st.chat_message("assistant"):
            st.write("관련된 정보를 찾을 수 없습니다. 다시 시도해주세요.")
