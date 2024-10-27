# from flask import Flask, jsonify, request
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import chromadb
import tempfile
import os,datetime,uuid
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
os.environ["USER_AGENT"] = "docSimilaroty/1.0"
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")
llm = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-pro")

def fetch_description_from_url(url):
    """Fetches job description text from a given URL."""
    loader = WebBaseLoader(url)
    job_description_data = loader.load().pop().page_content
    return job_description_data

def extract_job_profile(job_description_text):
    """Extracts job profile details from job description text."""
    prompt_template_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {Job_description_data}

        ### INSTRUCTION:
        The scraped text is from a career page. Extract the job postings and return `role`, `experience`, `skills`, and `description upto 50 words`.
        """
    )
    chain = prompt_template_extract | llm
    res = chain.invoke(input={"Job_description_data": job_description_text})
    # print("------------------")
    # print(res.content)
    # print("------------------")
    return res.content

def create_profile_embeddings(job_profile_fetched):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings=model.encode(job_profile_fetched)
    return embeddings
    
def load_resume_data(uploaded_files):
    """Processes uploaded resume files and loads data."""
    resume_data = []
    for upload_file in uploaded_files:
        if upload_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(upload_file.read())
                tmp_file_path = tmp_file.name
                loader = UnstructuredPDFLoader(tmp_file_path)
                data = loader.load()
                resume_data.append({"content": data, "metadata": {"source": data[0].metadata['source']}})
    return resume_data
    # for resume_path in uploaded_files:
    #     resume_loader = UnstructuredPDFLoader(resume_path)
    #     data = resume_loader.load()
    #     resume_data.append({"content": data, "metadata": {"source": resume_path}})
    # return resume_data

def create_resume_embeddings(job_resume_data):
    """Generates embeddings for each resume."""
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = [model.encode(resume['content'][0].page_content) for resume in job_resume_data]
    return embeddings

def store_embeddings_in_chroma(resume_data, embeddings):
    """Stores resume embeddings and metadata in ChromaDB."""
    chroma_client = chromadb.PersistentClient('vectorstore')
    # collection = chroma_client.get_or_create_collection(name="portfolio")
    collection = chroma_client.get_or_create_collection(name=f"portfolio_{str(uuid.uuid4())}")

    for idx, embedding in enumerate(embeddings):
        resume = resume_data[idx]
        collection.add(
            embeddings=embedding,
            metadatas=(resume['metadata']),
            documents=(resume['content'][0].page_content),
            ids=[str(uuid.uuid4())]
        )
    return collection

def match_job_resumes(job_profile_fetched,collection):
    """Queries ChromaDB to find matching resumes."""

    response = collection.query(
        # query_embeddings=job_profile_fetched,
        query_texts=job_profile_fetched,
        include=["documents", "metadatas","distances"],
        n_results=2
    )
    return response

def generate_profile_match_summary(final_match, job_profile_fetched):
    """Generates a summary comparing job profile to job description."""
    prompt_email = PromptTemplate.from_template(
        # """
        # ### JOB DESCRIPTION:
        # {job_description}

        # ### INSTRUCTION:
        # Write a short summary on how the profile matches with the job description. Here's the profile resume data: {job_profile}.
        # First, generate the name of the candidate from {job_profile}, then describe how the profile matches the job in about 100 words.
        # Include the resume file path at the end: {metadata}.
        # ### EMAIL (NO PREAMBLE):
        # """
        
        """
        ### JOB DESCRIPTION:
        {job_description}

        ### INSTRUCTION: This response will be passed to streamlit.write object generate based on that
        Write a short summary on how the profile matches with the job description. Here's the candidates resume data: {job_profile1}{job_profile2}.
        The response formate should be :
        **Candidate Name ** : Name of the candidate from resume data and display it in big font \n
        **Candidate skills ** : fetch skills from resume (dont add new line after each skill)
        **Skill that matches ** : matching skills with Job description \n
        **How he matches the Job description ** : Things which are matching
        **What stands out in his resume **: highilights from resume and how he matches the Job description in 50 words(dontcompare between resumes)\n
        Put a differenting line between each candidate
        Repeat the same formate for other candidate.who matches more should come first in response ,Include the resume file path at the end: {metadata}.
        ###(NO PREAMBLE):
        """

        # """
        # Prompt:
        # # ### JOB DESCRIPTION:
        # # {job_description}
        # Compare the two provided candidate profiles {job_profile} with the job description to determine the closest match. Format your response as follows, highlighting matching skills, distinctive strengths, and job alignment. Prioritize the candidate who aligns best with the job requirements
        # Response Format (Each candidate profile response should follow this format):
        # **Candidate Name **(Larger Font): [Insert candidate's name]
        # **Candidate Skills **: [List candidate's core skills from the resume]
        # **Matching Skills **: [List skills that match the job description]
        # **What Stands Out in Their Resume **: [Highlight unique achievements, qualifications, or experiences that make this candidate a strong match for the job, within 50 words]
        # Repeat the above structure for each candidate, placing the best match first. Attach the resume file path at the end, following: Resume File Path: [Insert path].
        # EMAIL (NO PREAMBLE):"""
    )
    chain_email = prompt_email | llm
    res = chain_email.invoke({
        "job_description": job_profile_fetched,
        "job_profile1": final_match["documents"][0][1],
        "job_profile2": final_match["documents"][0][0],
        "metadata": final_match["metadatas"]
    })
    st.write(res.content)
    # print("--------"*3)
    # print(final_match["documents"][0][0])
    # print("--------"*3)
    # print(final_match["documents"][0][1])

if __name__ == '__main__':
    st.title("Document Similarity Matcher")

    input_option = st.selectbox("Choose input method for Job Description", ("Enter URL", "Enter Text"))
    job_description_input = st.text_input("Enter the Job Description" if input_option == "Enter Text" else "Enter the Job Description URL")
    
    resume_files = st.file_uploader("Upload Resumes (PDF only)", type=["pdf"], accept_multiple_files=True)
    submit_button = st.button("Submit")

    if submit_button:
        if not job_description_input or not resume_files:
            st.error("Please provide a Job Description and upload at least one resume.")
        else:
            if input_option == "Enter URL":
                job_description_text = fetch_description_from_url(job_description_input)
            else:
                job_description_text = job_description_input
            
            # query _texts
            job_profile_fetched = extract_job_profile(job_description_text)
            job_resume_data = load_resume_data(resume_files)
            resumes_vector_embeddings = create_resume_embeddings(job_resume_data)
            vector_store_collection=store_embeddings_in_chroma(job_resume_data, resumes_vector_embeddings)
            final_match = match_job_resumes(job_profile_fetched,vector_store_collection)
            # print("final_match", final_match)
            (generate_profile_match_summary(final_match,job_profile_fetched))

            #query _embeddings
            # job_profile_fetched = extract_job_profile(job_description_text)
            # vector_store_descption= create_profile_embeddings(job_profile_fetched)
            # job_resume_data = load_resume_data(resume_files)
            # resumes_vector_embeddings = create_resume_embeddings(job_resume_data)
            # vector_store_collection=store_embeddings_in_chroma(job_resume_data, resumes_vector_embeddings)
            # final_match = match_job_resumes(vector_store_descption,vector_store_collection)
            # print("final_match", final_match)
            # generate_profile_match_summary(final_match,job_profile_fetched)
    
    # [14.64149509431969]
