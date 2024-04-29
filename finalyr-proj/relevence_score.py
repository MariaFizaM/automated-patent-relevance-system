import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Load the dataset
@st.cache_data
def load_data(query=None):
    df = pd.read_csv('neural_network_patent_query.csv')
    st.write('Data loaded')

    # Filter the dataset based on the query
    if query:
        df = df[df['patent_title'].str.contains(query, case=False)]

    # Preprocess the data
    dfs = df.sort_values(by='patent_number', ascending=True)
    dfs.set_index('patent_number', inplace=True)
    desired_columns = ['patent_title', 'patent_abstract', 'patent_date']
    dfs = dfs[desired_columns]
    dfs.loc[:, 'patent_title'] = dfs['patent_title'].str.lower()
    dfs.loc[:, 'patent_abstract'] = dfs['patent_abstract'].str.lower()
    df['patent_title'] = df['patent_title'].str.replace('The title of the patent is ', '', regex=True)
    dfs['text'] = "The title of the patent is " + dfs['patent_title'] + ' and its abstract is ' + dfs['patent_abstract'] + ' dated ' + dfs['patent_date']

    docs = dfs['text'].tolist()
    ids = [str(x) for x in dfs.index.tolist()]
    st.write('Data Preprocessed')
    return docs, ids

# Initialize the Chroma client and collection
@st.cache_resource
def initialize_chroma(docs,ids):
    client = chromadb.Client()
    collection = client.get_or_create_collection("patents")
    collection.add(documents=docs, ids=ids)
    st.write('Client and Collection Created')
    return client, collection

# Query the collection
def query_collection(collection, query):
    results = collection.query(query_texts=[query], n_results=8)
    return results['documents']

def extract_title(description):
    start_index = description.find("The title of the patent is ") + len("The title of the patent is ")
    end_index = description.find(" and its abstract is")
    return description[start_index:end_index]

def extract_abstract(description):
    start_index = description.find(" and its abstract is ") + len(" and its abstract is ")
    end_index = description.find(" dated")
    return description[start_index:end_index]

def extract_date(description):
    start_index = description.find("dated ") + len("dated ")

    return description[start_index:]

def evaluate_top_n_accuracy(query, retrieved_results, ground_truth):
    top_n = 3  # Define the value of N for Top-N Accuracy
    relevant_documents = ground_truth.get(query, [])[:top_n]  # Get relevant documents for the query
    retrieved_documents = [extract_title(result) for result in retrieved_results]  # Extract titles from retrieved results
    return any(doc in retrieved_documents for doc in relevant_documents)

def main():
    # Load ground truth data (Replace this with your ground truth dataset)
    ground_truth = {
        "query1": ["relevant_doc1", "relevant_doc2", "relevant_doc3"],
        "query2": ["relevant_doc2", "relevant_doc4", "relevant_doc5"],
        # Add more queries and relevant documents as needed
    }

    # Your existing code here
    st.set_page_config(
        page_title="Semantic Search with Chroma DB on Patents Dataset",
        page_icon=":blue_book:",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    st.markdown(f"<style>body {{ background-image: url('gradient_1.jpg'); background-repeat: no-repeat; background-size: cover; }}</style>", unsafe_allow_html=True)

    with open("design.css") as source_des:
        st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)

    st.title(" Semantic Search with Chroma DB on Patents Dataset")

    logo_path = "img2.jpeg"
    col1, col2, col3 = st.columns([4,2,5])

    with col1:
        st.write("")

    with col2:
        st.image(logo_path, width=300)  # Display the logo

    with col3:
        st.write("")

    # Get user input for the query
    query = st.text_input("Enter your query:")

    # Load the data
    docs, ids = load_data()

    # Initialize the Chroma client and collection
    client, collection = initialize_chroma(docs,ids)

    # Get user input for the query
    results = query_collection(collection, query)

    if query:
        # Query the collection
        results = query_collection(collection, query)

        if results:
            # Display the results
            for i, result in enumerate(results[:8]):
                # Your existing code for displaying results

            # Evaluate Top-N Accuracy
             top_n_accuracy = evaluate_top_n_accuracy(query, results, ground_truth)
            st.write(f"Top-N Accuracy for '{query}': {top_n_accuracy}")

        else:
            st.write("No results found for the query:", query)
    else:
        st.write("Please enter a query to search the patent collection.")

if __name__ == "__main__":
    main()