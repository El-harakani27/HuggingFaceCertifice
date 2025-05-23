import os
from datasets import load_dataset
from dotenv import load_dotenv
from langgraph.graph import START,END,StateGraph,MessagesState
from langchain_community.vectorstores import Chroma
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader,ArxivLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms import HuggingFaceEndpoint
from langchain.schema import Document
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
# dataset = load_dataset("gaia-benchmark/GAIA", "2023_all",trust_remote_code=True,data_dir='./data')
# val_dataset = dataset['validation']
# test_dataset = dataset['test']

# docs = []

# for i in val_dataset:
#   content= f"Question : {i['Question']}\nFinal answer : {i['Final answer']}"
#   doc = Document(page_content=content,
#                  metadata={
#                   "source":i['task_id']
#                   }
#                 )
#   docs.append(doc)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# vector_store = Chroma.from_documents(documents=docs,embedding=embeddings,persist_directory="./chromadb")
# vector_store.persist()
vector_store = Chroma(embedding_function=embeddings,persist_directory='./chromadb')
retriever = vector_store.as_retriever()

question_retrieve_tool = create_retriever_tool(
    retriever,
    name="Question Reriever",
    description="Find similar questions in the vector database for the given question"

)
@tool
def multiply(a:int,b:int)->int:
  """Multiply two numbers.
  Args:
    a: First int
    b: Second int
  Returns:
    Product of the two numbers
  """
  return a * b

@tool
def add(a:int,b:int)->int:
  """Adding two numbers.
  Args:
    a: First int
    b: Second int
  Returns:
    Sum of the two numbers
  """
  return a + b
@tool
def subtract(a:int,b:int)->int:
  """Subtracting two numbers.
  Args:
    a: First int
    b: Second int
  Returns:
    Difference of the two numbers
  """
  return a - b
@tool
def divide(a:int,b:int)->int:
  """Dividing two numbers.
  Args:
    a: First int
    b: Second int
  Returns:
    Division of the two numbers
  """
  return a / b 
@tool
def modulus(a:int,b:int)->int:
  """Modulus of two numbers.
  Args:
    a: First int
    b: Second int
  Returns:
    Modulus of the two numbers
  """
  return a % b
@tool
def wiki_search(query:str)->str:
  """Wikipedia search for a query and return 2 results.
    Args:
      query: Search query
    Returns:
      List of search results
  """
  wiki = WikipediaLoader(query=query,load_max_docs=2)
  serach_docs = wiki.load()
  formatted_search_docs = "\n\n---\n\n".join(
      [
          f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page","")}"/>\n{doc.page_content}\n</Document> ' 
          for doc in serach_docs
      ]
  )
  return {"web_results":formatted_search_docs}
@tool
def arvix_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 2 result.
    
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ])
    return {"arvix_results": formatted_search_docs}


tools = [
    multiply,
    divide,
    modulus,
    add,
    subtract,
    wiki_search,
    arvix_search
]


with open('./system_prompt.txt','r') as t:
  system_prompt = t.read()

sys_msg = SystemMessage(content=system_prompt)


def build_graph(llm_type:str = 'groq'):
  if llm_type == 'google':
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0)
  elif llm_type == 'groq':
    llm = ChatGroq(model="qwen-qwq-32b", temperature=0)
  else:
      raise ValueError(f"Invalid llm_type: {llm_type}")
  llm_with_tools = llm.bind_tools(tools)
  def assistant(state:MessagesState):
    """Assistant Node"""
    return {"messages":[llm_with_tools.invoke(state["messages"])]}
  def retriver_func(state:MessagesState):
    similar_question = vector_store.similarity_search(state["messages"][0].content)
    example_msg = HumanMessage(
        content = f"Here i provide a similar question and answer for refrence:\n\n{similar_question[0].page_content}"
    )
    return {"messages":[sys_msg] + state["messages"] + [example_msg]}
  builder = StateGraph(MessagesState)
  builder.add_node("retriever",retriver_func)
  builder.add_node("assistant",assistant)
  builder.add_node("tools",ToolNode(tools))
  builder.add_edge(START,"retriever")
  builder.add_edge("retriever","assistant")
  builder.add_conditional_edges(
      "assistant",
      tools_condition
  )
  builder.add_edge("tools","assistant")
  return builder.compile()

if __name__ == "__main__":
    question = "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia."
    # Build the graph
    graph = build_graph(llm_type="groq")
    # Run the graph
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})
    print(messages)
    for m in messages["messages"]:
        m.pretty_print()