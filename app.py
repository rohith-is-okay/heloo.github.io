from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import sys
import os
from flask import Flask, render_template, request,jsonify

os.environ["OPENAI_API_KEY"] = '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'
app = Flask(__name__)
def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    query = request.form['query']
    response = chatbot(query)  # Use the chatbot function to get the response
    return jsonify({"response": response})

index = construct_index("docs")

if __name__ == '__main__':
    app.run(debug=True)