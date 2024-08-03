import os
import json
import pandas as pd
from tqdm import tqdm
import ollama
import re

# Function to read JSON files and process them
def read_json_files(folder_path):
    data = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    data.append((file, json_data, root))
    return data

# Function to send text to OLLAMA and get QA pairs
def generate_qa(text, model="llama3"):
    prompt = f"Genera domande e risposte dai dati testuali italiani di InfoSapienza nel formato 1. domanda, risposta, 2. domanda, risposta, ecc.\n\n{text}\n\nFornisci tutte le risposte solo in italiano."
    response = ollama.generate(model=model, prompt=prompt)
    return response['response']

def process_qa(qa_text):
    pattern = re.compile(r'(\d+)[\.\)]\s*(?:Q:|Question:|Domanda:|What is|Qual è|)(.*?)(?:(?:\*\*|\s*|)\n*|\s+)\s*(?:A:|Answer:|Risposta:)\s*(.*?)(?=\n\d+[\.\)]|$)', re.DOTALL)
    # Find all matches in the text
    matches = pattern.findall(qa_text)
    # Format the matches into the desired list of lists, removing any '**' from questions and answers
    qa_pairs = [[re.sub(r'\*\*', '', q.strip()), re.sub(r'\*\*', '', a.strip())] for _, q, a in matches]
    
    return qa_pairs

# Function to check existing CSV and find processed chunks
def get_processed_chunks(csv_path):
    processed_chunks=[]
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            processed_chunks = df['chunk'].tolist()
        except:
            pass
    else:
        processed_chunks = []
    return set(processed_chunks)

def save_to_csv(output_data, csv_path):
    df = pd.DataFrame(output_data)
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

# Main function to process all folders and save the results
def main():
    folder_path = './preprocessed_text'
    csv_path = './output_qa_pairs.csv'
    processed_chunks = get_processed_chunks(csv_path)

    for folder in tqdm(os.listdir(folder_path), desc="Processing folders"):
        folder_full_path = os.path.join(folder_path, folder, 'clean_response') # pdf/clean_response
        if os.path.isdir(folder_full_path):
            json_data = read_json_files(folder_full_path) # read all json files

            # Process each file
            for file, data, root in tqdm(json_data, desc=f"Processing files in {folder}"):
                pdf_name = os.path.basename(os.path.dirname(root))
                page_number = file.split('_')[-2] if 'page' in file else None
                output_data = []

                if "chunks" in data:
                    for chunk in data["chunks"]:
                        if chunk not in processed_chunks:
                            qa_text = generate_qa(chunk)
                            qa_pairs = process_qa(qa_text)
                            for question, answer in qa_pairs:
                                output_data.append({
                                    "question": question,
                                    "answer": answer,
                                    "chunk": chunk,
                                    "title": None,
                                    "subheading": None,
                                    "pdf_name": pdf_name,
                                    "page_number": page_number
                                })
                            processed_chunks.add(chunk)

                elif "title" in data and "contents" in data:
                    for content in data["contents"]:
                        subheading = content.get("subheading", None)
                        text = content.get("text", "")
                        if text not in processed_chunks:
                            qa_text = generate_qa(text)
                            qa_pairs = process_qa(qa_text)
                            for question, answer in qa_pairs:
                                output_data.append({
                                    "question": question,
                                    "answer": answer,
                                    "chunk": text,
                                    "title": data["title"],
                                    "subheading": subheading,
                                    "pdf_name": pdf_name,
                                    "page_number": page_number
                                })
                            processed_chunks.add(text)

                # Save output data to CSV after processing each file
                if output_data:
                    save_to_csv(output_data, csv_path)

    print("CSV file has been saved successfully.")

if __name__ == "__main__":
    if os.path.exists('output_qa_pairs.csv'):
        x=0
        #df = pd.read_csv('output_qa_pairs.csv')
        #print("PDFs done: ", df.pdf_name.unique())
    else:
        print("No existing CSV file found.")
    main()
