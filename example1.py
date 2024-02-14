#Integrate code with openai api
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain #llmchains help in executing the prompt templates
from langchain.chains import SimpleSequentialChain, SequentialChain 

#For storing converstoion in memory buffer
from langchain.memory import ConversationBufferMemory

import streamlit as st #Less worry about uI with streamlit


os.environ["OPENAI_API_KEY"] = openai_key

#streamlit framework
st.title("Celebrity Search Results")
input_text = st.text_input("Search the topic about a celeb")


#Prompt templates
first_input_prompt = PromptTemplate(
    input_variables=["name"],
    template="Tell me about celebrity {name}"
)

#memory
person_memory = ConversationBufferMemory(input_key="name", memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key="person", memory_key='dob_history') 
events_memory = ConversationBufferMemory(input_key="dob", memory_key='event_history')
#Openai llms
#temp -> how much balance you want
llm = OpenAI(temperature=0.8)
#corresponding to every prompt template you will have llm chain
#Combining multiple prompt templates
chain = LLMChain(llm=llm, prompt = first_input_prompt, verbose=True, output_key = 'person', memory=person_memory) #specifically run this template

#Prompt templates
second_input_prompt = PromptTemplate(
    input_variables=["person"],
    template="When was {person} born?"
)
chain2 = LLMChain(llm=llm, prompt = second_input_prompt, verbose=True, output_key = 'dob', memory=dob_memory)

#Prompt templates
third_input_prompt = PromptTemplate(
    input_variables=["dob"],
    template="Mention 5 major events happened around {dob} in the world"
)
chain3 = LLMChain(llm=llm, prompt = third_input_prompt, verbose=True, output_key = 'events', memory=events_memory)

#combine chains
# parent_chain = SimpleSequentialChain(chains=[chain, chain2], verbose=True) #It shows only the last info
parent_chain= SequentialChain(
    chains=[chain,chain2, chain3], output_variables=["person", "dob", "events"], input_variables= ["name"], verbose=True) #It shows all the info


if input_text:
    temp = parent_chain({"name": input_text})
    print(temp)
    st.write(temp)
    # st.write(parent_chain.run(input_text))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('DOB'):
        st.info(dob_memory.buffer)

    with st.expander('Events'):
        st.info(events_memory.buffer)



