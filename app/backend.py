# Import necessary modules
import os
import base64
from flask import Flask, request, jsonify
from unstructured.partition.pdf import partition_pdf
import pytesseract
#from langchain.retrievers.multi_vector import MultiVectorRetriever # type: ignore
#from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
#from langchain.llms import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
#from langchain.retrievers import DenseRetriever
import google.generativeai as genai

# Configure Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Set paths
input_path = os.getcwd()
output_path = os.path.join(os.getcwd(), "output")

#if not os.path.exists(output_path):
    #os.makedirs(output_path)

# Initialize Flask app
app = Flask(__name__)

# Configure Google Generative AI
genai.configure(api_key="AIzaSyDQpLknCchBJYcW7BaIe87yHdhiKcqNvYA")

# Initialize Generative AI models
text_model = genai.GenerativeModel('models/gemini-pro')
table_model = genai.GenerativeModel('models/gemini-pro')
vision_model = genai.GenerativeModel('models/gemini-pro-vision')

# Initialize embedding model for langchain
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyDQpLknCchBJYcW7BaIe87yHdhiKcqNvYA")

# Initialize vector store and storage layer
vectorstore = Chroma(collection_name="summaries", embedding_function=embedding_model, persist_directory='./chroma_db')


# Function to process file and query
def process_file_and_query(file, query):
    # Partition PDF
    raw_pdf_elements = partition_pdf(filename=file, extract_images_in_pdf=True, infer_table_structure=True,
                                      chunking_strategy="by_title", max_characters=4000, new_after_n_chars=3800,
                                      combine_text_under_n_chars=2000, image_output_dir_path=output_path)

    # Process and encode elements
    text_elements = []
    table_elements = []
    image_elements = []

    # Function to encode images
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    for element in raw_pdf_elements:
        if 'CompositeElement' in str(type(element)):
            text_elements.append(element.text)
        elif 'Table' in str(type(element)):
            table_elements.append(element.text)

    def get_prediction(model_name, prompt):
        try:
            # Generate text using the specified model and prompt
            response = model_name.generate_content(prompt)

            # Assuming the response structure matches the expected format
            generated_text = response.text if response else "No generations found."

            return generated_text

        except Exception as e:
            print(f"An error occurred during text generation: {e}")
            return None
        
    def get_prediction_image(model_name, image):
        try:
            # Generate text using the specified model and prompt
            response = model_name.generate_content(image)

            # Assuming the response structure matches the expected format
            generated_text = response.text if response else "No generations found."

            return generated_text
        except Exception as e:
            print(f"An error occurred during text generation: {e}")
            return None


    # Encode images
    for image_file in os.listdir(output_path):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(output_path, image_file)
            encoded_image = encode_image(image_path)
            image_elements.append(encoded_image)

    def summarize_text(text_element):
        prompt = f"Summarize the following text:\n\n{text_element}\n\nSummary:"
        return get_prediction(text_model, prompt)

    def summarize_table(table_element):
        prompt = f"Summarize the following table:\n\n{table_element}\n\nSummary:"
        return get_prediction(table_model, prompt)

    def summarize_image(encoded_image):
        prompt = f"Describe the contents of the following image:\n\n{encoded_image}\n\nDescription:"
        return get_prediction_image(vision_model, prompt)

    table_summaries = [summarize_table(te) for te in table_elements[:2]]
    text_summaries = [summarize_text(te) for te in text_elements[:2]]
    image_summaries = [summarize_image(ie) for ie in image_elements]

    # Function to add documents to the retriever
    def add_documents_to_retriever(summaries, original_contents):
        doc_ids = [str(uuid.uuid4()) for _ in summaries]
        summary_docs = [ Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(summaries)]
        retriever.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, original_contents)))

    # Add text summaries
    add_documents_to_retriever(text_summaries, text_elements)

    # Add table summaries
    add_documents_to_retriever(table_summaries, table_elements)

    # Add image summaries
    add_documents_to_retriever(image_summaries, image_summaries)

    # Retrieve document example
    relevant_docs = retriever.retrieve(query)

    # Define a prompt template and use the model to answer questions based on the context
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA

    prompt_template = """Answer the question based only on the following context, which can include text, images, and tables:
    {context}
    Question: {question}
    """
    qa_chain = RetrievalQA(llm=text_model, retriever=retriever, prompt_template=prompt_template)

    response = qa_chain.run({"question": query})

    return response

# Route to handle file and query processing
@app.route('/process', methods=['POST'])
def process_request():
    if 'file' not in request.files or 'query' not in request.form:
        return jsonify({'response': 'No file or query provided'}), 400

    file = request.files['file']
    query = request.form['query']

    response = process_file_and_query(file, query)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(port=5001)

