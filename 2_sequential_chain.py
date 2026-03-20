import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# changing project name in environment
os.environ['LANGCHAIN_PROJECT'] = 'langsmith-seq-llm-project'

load_dotenv()


prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

config = {
    'run_name' : 'sequential_chain',
    'tags' : ['seq-llm', 'langsmith', 'trace'],
    'metadata' : {
        'model1' : 'gpt-4o-mini',
        'model2' : 'gpt-4o'
    }
}

model1 = ChatOpenAI(model='gpt-4o-mini')
model2 = ChatOpenAI(model='gpt-4o')

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

result = chain.invoke({'topic': 'Unemployment in America'})

print(result)
