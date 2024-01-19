import os

import bs4
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from operator import itemgetter
from langchain.prompts import PromptTemplate

from langchain.schema.runnable import RunnableMap
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import AIMessage, HumanMessage
from colorama import Fore, Back, Style
from sitemap import parse_urls

# getpass.getpass()
os.environ["OPENAI_API_KEY"] = "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"

urls = parse_urls()
print("URLs Length:", len(urls))

docs =[]
for url in urls:
    print(url,)
    string = url,
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                id=("layout-grid__area--maincontent")
            )
        ),
    )

    doc = loader.load()
    docs.append(doc[0])

print("Splitting docs: ", len(docs))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

splits = text_splitter.split_documents(docs)

print("Length All Splits", len(splits))
print("Length One Split", len(splits[0].page_content))
print("Storing splits in database")


vectorstore = Chroma.from_documents(
    documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

""" retrieved_docs = retriever.get_relevant_documents(
    "What are the approaches to Task Decomposition?"
)
print("Length Retrived Docs", len(retrieved_docs))
 """
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible.  
{context}
Question: {question}
Helpful Answer:"""
rag_prompt_custom = PromptTemplate.from_template(template)

rag_chain_from_docs = (
    {
        "context": lambda input: format_docs(input["documents"]),
        "question": itemgetter("question"),
    }
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableMap(
    {"documents": retriever, "question": RunnablePassthrough()}
) | {
    "documents": lambda input: [doc.metadata for doc in input["documents"]],
    "answer": rag_chain_from_docs,
}

""" print("Was ist ein Ankunftszentrum?", rag_chain.invoke("Was ist ein Ankunftszentrum?"))
print("Wie koche ich Hühnersuppe?", rag_chain.invoke("Wie koche ich Hühnersuppe?"))
print("Wie kann ich mein Auto anmelden?", rag_chain.invoke("Wie koche ich Hühnersuppe?"))

print("With Source: Ich bin in Berlin. Wo ist das Ankunftszentrum?", rag_chain_with_source.invoke("Ich bin in Berlin. Wo ist das Ankunftszentrum?")) """

condense_q_system_prompt = """Given a chat history and the latest user question \
which might reference the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
condense_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condense_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
condense_q_chain = condense_q_prompt | llm | StrOutputParser()

""" condense_q_chain.invoke(
    {
        "chat_history": [
            HumanMessage(content="What does LLM stand for?"),
            AIMessage(content="Large language model"),
        ],
        "question": "What is meant by large",
    }
)

condense_q_chain.invoke(
    {
        "chat_history": [
            HumanMessage(content="What does LLM stand for?"),
            AIMessage(content="Large language model"),
        ],
        "question": "How do transformers work",
    }
)
 """

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def condense_question(input: dict):
    if input.get("chat_history"):
        return condense_q_chain
    else:
        return input["question"]


rag_chain = (
    RunnablePassthrough.assign(
        context=condense_question | retriever | format_docs)
    | qa_prompt
    | llm
)

chat_history = []

""" question = "What is Task Decomposition?"

ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=question), ai_msg])

second_question = "What are common ways of doing it?"
rag_chain.invoke({"question": second_question, "chat_history": chat_history}) """

print("Local AuthoriChat. Type 'exit', to exit application.")
user_input = ''
while user_input != 'exit':
    user_input = input(Fore.BLUE + "Enter Question: ")
    ai_msg = rag_chain.invoke(
        {"question": user_input, "chat_history": chat_history})
    print(Fore.GREEN + ai_msg.content)
    chat_history.extend([HumanMessage(content=user_input), ai_msg])

vectorstore.delete_collection()

'''
def main():
    url = input("Enter URL: ")

    text_content = utils.parse_html_to_text(url)

    if text_content:
        print(text_content)
    else:
        print("Error: Unable to parse API documentation")
'''

""" if __name__ == '__main__':
    main()
 """
