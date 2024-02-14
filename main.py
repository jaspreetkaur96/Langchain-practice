#Integrate code with openai api
import os
from constants import openai_key
from langchain.llms import OpenAI

import streamlit as st #Less worry about uI with streamlit


os.environ["OPENAI_API_KEY"] = openai_key

#streamlit framework
st.title("LangChain Demo with OpenAI api")
input_text = st.text_input("Search the topic you want")

#Openai llms
#temp -> how much balance you want
llm = OpenAI(temperature=0.8)


if input_text:
    st.write(llm(input_text))



