import openai
import os
import base64
from openai import OpenAI
from unstructured.partition.pdf import partition_pdf
from IPython.display import Image, display
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import uuid
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from base64 import b64decode
from langchain.chains import LLMChain

# OPENAI_API_KEY = "" 
# OPENAI_API_KEY2 = ""

poppler_path = r"C:/poppler-24.08.0/Library/bin"  


os.environ["PATH"] = poppler_path

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = ''
os.environ["GROQ_API_KEY"] = ""
# os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def extract_chunks_from_pdf(file_path, max_characters=10000, combine_text_under_n_chars=2000, new_after_n_chars=6000):
    """
    Extracts chunks from a PDF file using the Unstructured library.

    Params:
        file_path (str): The path to the PDF file.
        max_characters (int): Maximum characters in a chunk (default: 10000).
        combine_text_under_n_chars (int): Combine text if under this limit (default: 2000).
        new_after_n_chars (int): Create new chunks after this many characters (default: 6000).

    Returns:
        list: A list of extracted chunks containing text, tables, and images.
    """
    try:
        chunks = partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=max_characters,
            combine_text_under_n_chars=combine_text_under_n_chars,
            new_after_n_chars=new_after_n_chars,
        )
        return chunks
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []


def separate_chunks(chunks):
    """
    Separates tables and text elements from a list of chunks.

    Params:
        chunks (list): A list of chunks (elements) extracted from a PDF.

    Returns:
        tuple: A tuple containing two lists:
            - tables: List of table elements.
            - texts: List of text elements.
    """
    tables = []
    texts = []
    try:
        for chunk in chunks:
            if "Table" in str(type(chunk)):
                tables.append(chunk)
            if "CompositeElement" in str(type(chunk)):
                texts.append(chunk)
        return tables, texts
    except Exception as e:
        print(f"Error separating chunks: {e}")


def extract_images_from_chunk(chunk):
    """
    Extracts images from a specific chunk's metadata original elements.

    Args:
        chunk (object): A chunk object containing metadata and original elements.

    Returns:
        list: A list of image elements (if any) from the chunk.
    """
    try:
        elements = chunk.metadata.orig_elements
        chunk_images = [el for el in elements if "Image" in str(type(el))]
        return chunk_images
    except AttributeError:
        print("No images found or chunk does not contain metadata.orig_elements.")
        return []


def get_images_base64(chunks):
    """
    Get base64-encoded images from CompositeElement objects.

    Args:
        chunks (list): List of chunks.

    Returns:
        list: Base64-encoded image strings.
    """
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64


def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

def summarize_texts(texts, model):
    """
    Summarizes a list of text chunks using the specified model.

    Args:
        texts (list): List of text chunks to summarize.
        model (ChatOpenAI): The language model instance for summarization.

    Returns:
        list: Summaries of the text chunks.
    """
    prompt_text = """
    You are an assistant tasked with analyzing research articles. Your task is to:
    1. Summarize the key findings of the research article concisely.
    2. Extract and describe any numeric data, graphs, or tables presented in the article.
    Provide the summary and extracted details in a structured format.

    Table or text chunk: {element}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    return summarize_chain.batch(texts[:10], {"max_concurrency": 3})

def summarize_tables(tables, model):
    """
    Summarizes a list of table elements using the specified model.

    Args:
        tables (list): List of table elements to summarize (metadata should contain 'text_as_html').
        model (ChatOpenAI): The language model instance for summarization.

    Returns:
        list: Summaries of the table elements.
    """
    # Extract HTML representations of the tables
    tables_html = [table.metadata.text_as_html for table in tables]

    # Define the prompt template
    prompt_text = """
      You are an assistant tasked with analyzing research articles. Your task is to:
      1. Summarize the key findings of the research article concisely.
      2. Extract and describe any numeric data, graphs, or tables presented in the article.
      Provide the summary and extracted details in a structured format.

    Table or text chunk: {element}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Define the summarization chain
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Summarize the tables
    return summarize_chain.batch(tables_html, {"max_concurrency": 3})

def summarize_images(images,model):
    """
    Generates detailed descriptions of images using a pre-defined prompt.

    Args:
        images (list): A list of base64-encoded image strings.

    Returns:
        list: Summaries of the images.
    """
    # Define the prompt template
    prompt_template = """Describe the image in detail. Be specific about graphs, such as bar plots."""
    
    # Define messages for the prompt
    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]

    # Create the prompt
    prompt = ChatPromptTemplate.from_messages(messages)

    # Define the chain
    chain = prompt | ChatOpenAI(model=model) | StrOutputParser()

    # Generate summaries for the images
    return chain.batch(images)

def initialize_multi_modal_retriever(collection_name="multi_modal_rag", embedding_model=None):
    """
    Initializes a multi-modal retriever using Chroma for child chunks and InMemoryStore for parent documents.

    Args:
        collection_name (str): Name of the collection in the vector store (default: "multi_modal_rag").
        embedding_model (object): Embedding function to use for the vector store (default: OpenAIEmbeddings).

    Returns:
        MultiVectorRetriever: The initialized multi-vector retriever.
    """
    embedding_model = embedding_model or OpenAIEmbeddings()
    vectorstore = Chroma(collection_name=collection_name, embedding_function=embedding_model)
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    return retriever

def add_data_to_retriever(
    retriever,
    texts=None,
    text_summaries=None,
    tables=None,
    table_summaries=None,
    images=None,
    image_summaries=None,
    id_key="doc_id"
):
    """
    Adds text, table, and image data to the multi-modal retriever.

    Args:
        retriever (MultiVectorRetriever): The initialized retriever.
        texts (list): Original text data.
        text_summaries (list): Summaries of the text data.
        tables (list): Original table data.
        table_summaries (list): Summaries of the table data.
        images (list): Original image data (e.g., base64 strings or URLs).
        image_summaries (list): Summaries of the image data.
    """
    if texts and text_summaries:
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=summary, metadata={id_key: doc_ids[i]})
            for i, summary in enumerate(text_summaries)
        ]
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, texts)))

    if tables and table_summaries:
        table_ids = [str(uuid.uuid4()) for _ in tables]
        summary_tables = [
            Document(page_content=summary, metadata={id_key: table_ids[i]})
            for i, summary in enumerate(table_summaries)
        ]
        retriever.vectorstore.add_documents(summary_tables)
        retriever.docstore.mset(list(zip(table_ids, tables)))

    if images and image_summaries:
        img_ids = [str(uuid.uuid4()) for _ in images]
        summary_images = [
            Document(page_content=summary, metadata={id_key: img_ids[i]})
            for i, summary in enumerate(image_summaries)
        ]
        retriever.vectorstore.add_documents(summary_images)
        retriever.docstore.mset(list(zip(img_ids, images)))

def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {"images": b64, "texts": text}

def build_prompt(kwargs):

    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text

    # construct prompt with context (including images)
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )





# Example usage of functions defined in Task2

# Step 1: Process a PDF file
file_path = '../pdfs/Developing an LLM-Powered Document Analyzer.pdf'  # Replace with your PDF file path
chunks = extract_chunks_from_pdf(
    file_path,
    max_characters=10000,
    combine_text_under_n_chars=2000,
    new_after_n_chars=6000
)

# Step 2: Print the first few chunks for inspection
print(f"Extracted {len(chunks)} chunks from the PDF.")
# for idx, chunk in enumerate(chunks[:3]):  # Display the first 3 chunks
#     print(f"Chunk {idx + 1}:")
#     print(chunk)
#     print("-" * 40)

# Step 3: Separate the chunks into tables and texts
tables, texts = separate_chunks(chunks)
print(f"Extracted {len(tables)} tables and {len(texts)} text chunks.")

# Step 4: Extract images from chunks
chunk_images = extract_images_from_chunk(chunks)
images = get_images_base64(chunks)
print(f"Extracted {len(images)} images.")

# Step 5: Display an image (optional)
# if images:
#     display_base64_image(images[0])  # Display the first image (if any)

model = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")

# Step 7: Summarize text chunks
if texts:
    text_summaries = summarize_texts(texts, model)
    print(f"Generated summaries for {len(text_summaries)} text chunks.")
    # for idx, summary in enumerate(text_summaries[:3]):  # Show first 3 summaries
    #     print(f"Text {idx + 1} Summary:")
    #     print(summary)
    #     print("-" * 40)

# Step 8: Summarize table elements
if tables:
    table_summaries = summarize_tables(tables, model)
    print(f"Generated summaries for {len(table_summaries)} table elements.")
    # for idx, summary in enumerate(table_summaries[:3]):  # Show first 3 summaries
    #     print(f"Table {idx + 1} Summary:")
    #     print(summary)
    #     print("-" * 40)

# Step 9: Summarize images
if images:
    image_summaries = summarize_images(images, model)
    print(f"Generated summaries for {len(image_summaries)} images.")
    # for idx, summary in enumerate(image_summaries[:3]):  # Show first 3 summaries
    #     print(f"Image {idx + 1} Summary:")
    #     print(summary)
    #     print("-" * 40)

# Step 10: Initialize a multi-modal retriever
retriever = initialize_multi_modal_retriever()

# Step 11: Add summarized data to the retriever
add_data_to_retriever(
    retriever,
    texts=texts,
    text_summaries=text_summaries if texts else None,
    tables=tables,
    table_summaries=table_summaries if tables else None,
    images=images,
    image_summaries=image_summaries if images else None
)

# Step 12: Query the retriever
response = retriever.invoke("What are the key findings in this document?")
print("Response from the retriever:")
print(response)
