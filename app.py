#load libraries
import streamlit as st
import yfinance as yf
from finmath_model import answer_question  

st.set_page_config(page_title="FinMath-LLM", layout="centered")
st.title("FinMath-LLM: Financial Math Tutor")

#Sidebar 
menu = st.sidebar.selectbox(
    "Menu",
    ["Math Tutor", "Live Stock Price", "About App"]
)

#Math tutor 
if menu == "Math Tutor":
    st.markdown("Ask any **financial math** question and get a step-by-step explanation.")
    question = st.text_input("Enter your finance/math question:",
                             placeholder="Example: What is the compound interest on $1000 at 5% for 10 years?")
    
    if question:
        with st.spinner("Solving with FinMath..."):
            result = answer_question(question)
        st.text_area("Answer:", result, height=300)

#Live stock price 
elif menu == "Live Stock Price":
    st.markdown("Get real-time stock prices using Yahoo Finance.")
    ticker = st.text_input("Enter stock ticker (example, AAPL, TSLA):")

    if ticker:
        stock = yf.Ticker(ticker.upper())
        data = stock.history(period="7d", interval="1h")  

        if not data.empty:
            latest_price = data["Close"].iloc[-1]
            st.metric(label=f"{ticker.upper()} Latest Price", value=f"${latest_price:.2f}")
            st.line_chart(data["Close"])
        else:
            st.warning("No data found. Check the ticker symbol.")

#About page
elif menu == "About App":
    st.subheader("About FinMath-GPT")
    st.markdown("""
    
    **FinMath-LLM** is your AI-powered financial math tutor — designed to help students, professionals, and learners solve financial problems with clear, step-by-step explanations.

    ### What It Does:
    - Understands your finance/math questions in plain English
    - Solves problems like loan payments, interest, NPV, and more
    - Displays real-time stock prices using Yahoo Finance
    - Provides accurate and easy-to-read solutions with explanations

    ### Who It's For:
    - Students preparing for finance, economics, or accounting exams
    - Professionals needing quick financial insights
    - Self-learners exploring personal finance or investment topics

    ### How It Works:
    FinMath-LLM uses advanced language models (LLMs) to understand your queries and return structured, tutor-style responses using real mathematical logic and formulas.

    ---
    Developed by Nolvenne Tah to make financial education more accessible and interactive through AI.  
    Questions or feedback? Reach out at **nolvenetah@gmail.com**
    """,
    unsafe_allow_html=True
    )
