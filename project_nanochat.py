import os
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
import numpy as np
import faiss
import pickle
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
EMBEDDINGS_DIRECTORY = "embeddings"  # Separate folder for FAISS indexes
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
OUTPUT_CSV = "qa_dataset.csv"

# Create embeddings directory if it doesn't exist
if not os.path.exists(EMBEDDINGS_DIRECTORY):
    os.makedirs(EMBEDDINGS_DIRECTORY)

# --- App Title and Sidebar ---
st.set_page_config(page_title="Q&A Dataset Generator", layout="wide")
st.title("📄 Q&A Dataset Generator with Ollama & FAISS")
st.markdown("This tool uses embeddings and semantic similarity to create contextually-aware Q&A pairs from PDFs.")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # PDF Directory Selection
    st.subheader("📁 PDF Folder Selection")
    
    # Get list of directories in current folder
    current_dir = os.getcwd()
    available_dirs = [d for d in os.listdir(current_dir) 
                     if os.path.isdir(os.path.join(current_dir, d)) 
                     and not d.startswith('.')]
    
    # Add option for custom path
    folder_selection_mode = st.radio(
        "Select folder mode:",
        ["Choose from available folders", "Enter custom path"],
        help="Choose a folder from the list or enter a custom path"
    )
    
    if folder_selection_mode == "Choose from available folders":
        if available_dirs:
            # Add option to create new folder
            folder_options = ["[Create New Folder]"] + sorted(available_dirs)
            selected_folder = st.selectbox(
                "Select PDF folder:",
                options=folder_options,
                index=1 if len(folder_options) > 1 else 0
            )
            
            if selected_folder == "[Create New Folder]":
                new_folder_name = st.text_input("Enter new folder name:", "pdfs")
                if st.button("Create Folder"):
                    new_folder_path = os.path.join(current_dir, new_folder_name)
                    if not os.path.exists(new_folder_path):
                        os.makedirs(new_folder_path)
                        st.success(f"✅ Created folder: {new_folder_name}")
                        st.rerun()
                    else:
                        st.warning(f"Folder '{new_folder_name}' already exists!")
                PDF_DIRECTORY = new_folder_name  # Temporary, will be created
            else:
                PDF_DIRECTORY = selected_folder
        else:
            st.warning("No folders found in current directory.")
            new_folder_name = st.text_input("Create a new folder:", "pdfs")
            if st.button("Create Folder"):
                if new_folder_name:
                    os.makedirs(new_folder_name, exist_ok=True)
                    st.success(f"✅ Created folder: {new_folder_name}")
                    st.rerun()
            PDF_DIRECTORY = new_folder_name
    else:
        # Custom path input
        custom_path = st.text_input(
            "Enter PDF folder path:",
            value="pdfs2",
            help="Enter relative or absolute path to your PDF folder"
        )
        PDF_DIRECTORY = custom_path
    
    # Check if directory exists
    if not os.path.exists(PDF_DIRECTORY):
        st.error(f"⚠️ The folder '{PDF_DIRECTORY}' does not exist!")
        if st.button("Create This Folder"):
            os.makedirs(PDF_DIRECTORY, exist_ok=True)
            st.success(f"✅ Created folder: {PDF_DIRECTORY}")
            st.rerun()
        st.info("Please create the folder and add your PDF files, then refresh the page.")
        st.stop()
    else:
        st.success(f"✅ Using folder: `{PDF_DIRECTORY}`")
    
    # Show folder path
    abs_path = os.path.abspath(PDF_DIRECTORY)
    st.caption(f"📂 Full path: `{abs_path}`")
    
    st.divider()
    
    # Model configuration
    st.subheader("🤖 Model Settings")
    ollama_model = st.text_input("Enter Ollama Model Name", "gemma3:4b")
    
    # Embedding configuration
    st.subheader("🔢 Embedding Settings (FAISS)")
    st.info("⚠️ Embeddings are REQUIRED for Q&A generation")
    embedding_model = st.text_input("Embedding Model", "qwen3-embedding:0.6b")
    
    # Context settings
    st.subheader("🎯 Context Settings")
    num_related_chunks = st.slider("Related chunks for context", 0, 5, 3, 
                                   help="Number of semantically similar chunks to use as context")
    
    st.caption(f"Embeddings stored in: ./{EMBEDDINGS_DIRECTORY}/")
    
    st.divider()
    
    # Count PDFs
    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith('.pdf')]
    st.metric("PDFs Found", len(pdf_files))
    
    # Count existing FAISS indexes
    if os.path.exists(EMBEDDINGS_DIRECTORY):
        faiss_files = [f for f in os.listdir(EMBEDDINGS_DIRECTORY) if f.endswith('.faiss')]
        st.metric("FAISS Indexes", len(faiss_files))
    
    if len(pdf_files) > 0:
        st.write("**Files to process:**")
        for pdf in pdf_files:
            st.text(f"• {pdf}")
    else:
        st.warning("No PDF files found in selected folder!")
        st.info("Add PDF files to the folder and refresh.")

# --- Helper Functions ---
def extract_and_chunk_pdf(file_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Extract text from PDF and split into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    doc = fitz.open(file_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    
    metadata = {"source": os.path.basename(file_path)}
    splits = text_splitter.create_documents([text], metadatas=[metadata])
    
    return splits

def generate_qa_from_chunk_with_context(main_chunk, related_chunks, llm):
    """Generate Q&A using main chunk and semantically related chunks for better context"""
    
    # Combine main chunk with related chunks for context
    context = f"MAIN TEXT:\n{main_chunk}\n\n"
    
    if related_chunks:
        context += "RELATED CONTEXT (semantically similar sections):\n"
        for i, chunk in enumerate(related_chunks, 1):
            context += f"{i}. {chunk}\n\n"
    
    qa_prompt_template = """Based on the following main text and related context, generate ONE relevant question and its answer. 

The RELATED CONTEXT contains semantically similar sections from the document that provide additional context.

INSTRUCTIONS:
1. Focus primarily on the MAIN TEXT for the question
2. Use RELATED CONTEXT to provide more comprehensive and contextual answers
3. Generate a clear, specific question that tests understanding of key concepts
4. Provide a comprehensive answer that leverages both main text and related context
5. Format your response EXACTLY as follows (no extra text):

QUESTION: [Your question here]
ANSWER: [Your answer here]

{context}

OUTPUT:"""

    prompt = PromptTemplate.from_template(qa_prompt_template)
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({"context": context})
        
        # Parse the response
        lines = response.strip().split('\n')
        question = ""
        answer = ""
        
        for line in lines:
            if line.startswith("QUESTION:"):
                question = line.replace("QUESTION:", "").strip()
            elif line.startswith("ANSWER:"):
                answer = line.replace("ANSWER:", "").strip()
        
        if not question or not answer:
            parts = response.split("ANSWER:")
            if len(parts) == 2:
                question = parts[0].replace("QUESTION:", "").strip()
                answer = parts[1].strip()
        
        return question, answer
    
    except Exception as e:
        st.warning(f"Error generating Q&A: {e}")
        return None, None

def generate_embedding(text, embeddings_model):
    """Generate embedding vector for text"""
    try:
        return embeddings_model.embed_query(text)
    except Exception as e:
        st.warning(f"Error generating embedding: {e}")
        return None

def get_faiss_filenames(pdf_filename):
    """Generate FAISS index and metadata filenames for a specific PDF"""
    base_name = os.path.splitext(pdf_filename)[0]
    faiss_index_path = os.path.join(EMBEDDINGS_DIRECTORY, f"{base_name}.faiss")
    metadata_path = os.path.join(EMBEDDINGS_DIRECTORY, f"{base_name}_metadata.pkl")
    return faiss_index_path, metadata_path

def delete_pdf_faiss_index(pdf_filename):
    """Delete FAISS index and metadata for a specific PDF"""
    faiss_path, metadata_path = get_faiss_filenames(pdf_filename)
    deleted = False
    
    if os.path.exists(faiss_path):
        os.remove(faiss_path)
        deleted = True
    
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
        deleted = True
    
    return deleted

def build_temp_faiss_index(embeddings_list):
    """Build a temporary FAISS index for finding similar chunks during Q&A generation"""
    if not embeddings_list:
        return None
    
    embeddings_array = np.array(embeddings_list).astype('float32')
    dimension = embeddings_array.shape[1]
    
    # Create FAISS index (using L2 distance)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    return index

def find_similar_chunks(query_idx, faiss_index, chunks, k=5):
    """Find k most similar chunks to the query chunk using FAISS"""
    if faiss_index is None or query_idx >= faiss_index.ntotal:
        return []
    
    # Get the query vector from the index
    query_vector = faiss_index.reconstruct(query_idx)
    query_vector = np.array([query_vector]).astype('float32')
    
    # Search for similar vectors (k+1 because the first result will be itself)
    distances, indices = faiss_index.search(query_vector, min(k + 1, faiss_index.ntotal))
    
    # Get similar chunks (excluding the query chunk itself)
    similar_chunks = []
    for idx in indices[0]:
        if idx != query_idx and idx < len(chunks):
            similar_chunks.append(chunks[idx].page_content)
    
    return similar_chunks[:k]

def save_pdf_faiss_index(pdf_filename, embeddings_list, qa_indices):
    """Save embeddings as FAISS index for a specific PDF"""
    if not embeddings_list:
        return None, None
    
    faiss_path, metadata_path = get_faiss_filenames(pdf_filename)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings_list).astype('float32')
    
    # Get dimension
    dimension = embeddings_array.shape[1]
    
    # Create FAISS index (using L2 distance)
    index = faiss.IndexFlatL2(dimension)
    
    # Add vectors to index
    index.add(embeddings_array)
    
    # Save FAISS index
    faiss.write_index(index, faiss_path)
    
    # Save metadata (QA indices and other info)
    metadata = {
        'qa_indices': qa_indices,
        'dimension': dimension,
        'total_vectors': len(embeddings_list)
    }
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    return faiss_path, metadata_path

def load_pdf_faiss_index(pdf_filename):
    """Load FAISS index and metadata for a specific PDF"""
    faiss_path, metadata_path = get_faiss_filenames(pdf_filename)
    
    if not os.path.exists(faiss_path) or not os.path.exists(metadata_path):
        return None, None
    
    try:
        # Load FAISS index
        index = faiss.read_index(faiss_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return index, metadata
    
    except Exception as e:
        st.warning(f"Error loading FAISS index: {e}")
        return None, None

def search_faiss_index(index, metadata, query_embedding, top_k=10):
    """Search FAISS index for similar vectors"""
    query_vector = np.array([query_embedding]).astype('float32')
    
    # Search index
    distances, indices = index.search(query_vector, min(top_k, index.ntotal))
    
    # Map to QA indices
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadata['qa_indices']):
            qa_idx = metadata['qa_indices'][idx]
            # Convert L2 distance to similarity score (inverse)
            similarity = 1 / (1 + dist)
            results.append((similarity, qa_idx))
    
    return results

# --- Main Application Logic ---
st.header("📚 Generate Q&A Dataset from PDFs")
st.markdown("""
This process will:
1. **Extract and chunk** each PDF
2. **Generate embeddings** for all chunks
3. **Build FAISS index** for semantic similarity
4. **Generate Q&A pairs** using embeddings to find related context
5. **Save FAISS indexes** for future semantic search
6. Save results to CSV with columns: `file`, `question`, `answer`, `context_chunks_used`
""")

st.info("🎯 **Key Feature:** Q&A generation uses semantic similarity to find related chunks, providing better context for answers!")

# Show current folder info
col1, col2 = st.columns(2)
with col1:
    st.info(f"📁 **Current PDF Folder:** `{PDF_DIRECTORY}`")
with col2:
    pdf_count = len([f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith('.pdf')]) if os.path.exists(PDF_DIRECTORY) else 0
    st.info(f"📄 **PDF Files:** {pdf_count}")

if st.button("🚀 Start Processing All PDFs", type="primary"):
    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        st.error("No PDF files found in the directory.")
    else:
        # Initialize LLM
        llm = Ollama(model=ollama_model)
        
        # Initialize embeddings model
        st.info(f"Initializing embedding model: {embedding_model}")
        embeddings_model = OllamaEmbeddings(model=embedding_model)
        
        # Initialize or load existing CSV
        if os.path.exists(OUTPUT_CSV):
            existing_df = pd.read_csv(OUTPUT_CSV)
            st.info(f"Found existing CSV with {len(existing_df)} entries. Will append new data.")
            all_qa_data = existing_df.to_dict('records')
        else:
            all_qa_data = []
        
        # Process each PDF
        total_pdfs = len(pdf_files)
        overall_progress = st.progress(0)
        status_text = st.empty()
        
        for pdf_idx, pdf_file in enumerate(pdf_files):
            status_text.markdown(f"### Processing PDF {pdf_idx + 1}/{total_pdfs}: **{pdf_file}**")
            
            file_path = os.path.join(PDF_DIRECTORY, pdf_file)
            
            # Delete old FAISS index for this PDF before processing
            deleted = delete_pdf_faiss_index(pdf_file)
            if deleted:
                st.info(f"🗑️ Deleted old FAISS index for {pdf_file}")
            else:
                st.info(f"✨ Creating fresh FAISS index for {pdf_file}")
            
            try:
                # Step 1: Extract and chunk the PDF
                with st.spinner(f"Extracting and chunking {pdf_file}..."):
                    chunks = extract_and_chunk_pdf(file_path)
                    st.success(f"✅ Created {len(chunks)} chunks from {pdf_file}")
                
                # Step 2: Generate embeddings for ALL chunks first
                st.info("🔢 Generating embeddings for all chunks...")
                embedding_progress = st.progress(0)
                
                chunk_embeddings = []
                for chunk_idx, chunk in enumerate(chunks):
                    embedding = generate_embedding(chunk.page_content, embeddings_model)
                    if embedding is not None:
                        chunk_embeddings.append(embedding)
                    else:
                        # If embedding fails, use zero vector
                        st.warning(f"Failed to generate embedding for chunk {chunk_idx}, skipping...")
                    
                    embedding_progress.progress((chunk_idx + 1) / len(chunks))
                
                embedding_progress.empty()
                st.success(f"✅ Generated {len(chunk_embeddings)} embeddings")
                
                # Step 3: Build temporary FAISS index for similarity search
                st.info("🏗️ Building temporary FAISS index for similarity search...")
                temp_faiss_index = build_temp_faiss_index(chunk_embeddings)
                st.success("✅ FAISS index built")
                
                # Step 4: Generate Q&A using embeddings for context
                st.info(f"💬 Generating context-aware Q&A pairs (using top {num_related_chunks} similar chunks)...")
                qa_progress = st.progress(0)
                qa_status = st.empty()
                
                pdf_embeddings = []
                pdf_qa_indices = []
                
                for chunk_idx, chunk in enumerate(chunks):
                    qa_status.text(f"Generating Q&A for chunk {chunk_idx + 1}/{len(chunks)} with semantic context...")
                    
                    # Find similar chunks using FAISS
                    related_chunks = find_similar_chunks(
                        chunk_idx, 
                        temp_faiss_index, 
                        chunks, 
                        k=num_related_chunks
                    )
                    
                    # Generate Q&A with context
                    question, answer = generate_qa_from_chunk_with_context(
                        chunk.page_content, 
                        related_chunks, 
                        llm
                    )
                    
                    if question and answer:
                        # Add to all_qa_data
                        qa_index = len(all_qa_data)
                        all_qa_data.append({
                            "file": pdf_file,
                            "question": question,
                            "answer": answer,
                            "context_chunks_used": len(related_chunks)
                        })
                        
                        # Store embedding for final FAISS index
                        if chunk_idx < len(chunk_embeddings):
                            pdf_embeddings.append(chunk_embeddings[chunk_idx])
                            pdf_qa_indices.append(qa_index)
                    
                    qa_progress.progress((chunk_idx + 1) / len(chunks))
                
                qa_progress.empty()
                qa_status.empty()
                
                # Step 5: Save to CSV after each PDF
                df = pd.DataFrame(all_qa_data)
                df.to_csv(OUTPUT_CSV, index=False)
                
                # Step 6: Save permanent FAISS index for this PDF
                if pdf_embeddings:
                    faiss_path, metadata_path = save_pdf_faiss_index(
                        pdf_file, 
                        pdf_embeddings, 
                        pdf_qa_indices
                    )
                    st.success(f"💾 Saved FAISS index with {len(pdf_embeddings)} vectors for {pdf_file}")
                    st.caption(f"Index: {faiss_path}")
                
                st.success(f"✅ Completed {pdf_file}. Total Q&A pairs: {len(all_qa_data)}")
                
            except Exception as e:
                st.error(f"❌ Failed to process {pdf_file}: {e}")
                import traceback
                st.code(traceback.format_exc())
            
            # Update overall progress
            overall_progress.progress((pdf_idx + 1) / total_pdfs)
        
        # Final summary
        status_text.markdown("### ✅ All PDFs Processed!")
        st.balloons()
        
        st.success(f"Generated {len(all_qa_data)} question-answer pairs from {total_pdfs} PDFs")
        
        # Count total FAISS indexes
        faiss_files = [f for f in os.listdir(EMBEDDINGS_DIRECTORY) if f.endswith('.faiss')]
        st.success(f"Created {len(faiss_files)} FAISS indexes (one per PDF)")
        
        # Display sample results
        with st.expander("📊 View Sample Q&A Pairs"):
            sample_df = pd.DataFrame(all_qa_data[-10:])  # Show last 10
            st.dataframe(sample_df)
        
        # Download button
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Complete Q&A Dataset (CSV)",
            data=csv_data,
            file_name=OUTPUT_CSV,
            mime="text/csv",
        )

# --- Display existing dataset if available ---
if os.path.exists(OUTPUT_CSV):
    st.divider()
    st.header("📊 Existing Q&A Dataset")
    
    df = pd.read_csv(OUTPUT_CSV)
    st.metric("Total Q&A Pairs", len(df))
    
    # Show statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Unique Files", df['file'].nunique())
    with col2:
        if df['file'].nunique() > 0:
            st.metric("Average Q&A per File", round(len(df) / df['file'].nunique(), 1))
    with col3:
        # Count FAISS indexes
        if os.path.exists(EMBEDDINGS_DIRECTORY):
            faiss_files = [f for f in os.listdir(EMBEDDINGS_DIRECTORY) if f.endswith('.faiss')]
            st.metric("FAISS Indexes", len(faiss_files))
    with col4:
        if 'context_chunks_used' in df.columns:
            avg_context = df['context_chunks_used'].mean()
            st.metric("Avg Context Chunks", f"{avg_context:.1f}")
    
    # Show breakdown by file
    with st.expander("View Q&A Count by File"):
        file_counts = df['file'].value_counts().reset_index()
        file_counts.columns = ['File', 'Q&A Pairs']
        
        # Add FAISS index status
        def has_faiss_index(filename):
            faiss_path, _ = get_faiss_filenames(filename)
            return "✅" if os.path.exists(faiss_path) else "❌"
        
        file_counts['Has FAISS Index'] = file_counts['File'].apply(has_faiss_index)
        st.dataframe(file_counts)
    
    # Show recent entries
    with st.expander("View Recent Q&A Pairs"):
        st.dataframe(df.tail(20))
    
    # Search functionality
    with st.expander("🔍 Search Q&A Pairs"):
        search_term = st.text_input("Search in questions or answers:")
        if search_term:
            mask = df['question'].str.contains(search_term, case=False, na=False) | \
                   df['answer'].str.contains(search_term, case=False, na=False)
            search_results = df[mask]
            st.write(f"Found {len(search_results)} results:")
            st.dataframe(search_results)
    
    # Semantic search with FAISS
    with st.expander("🔍 Semantic Search (Using FAISS)"):
        st.markdown("⚡ Fast semantic search using FAISS vector similarity")
        
        # Select which PDF to search in
        unique_files = sorted(df['file'].unique())
        
        # Check which files have FAISS indexes
        files_with_faiss = []
        for f in unique_files:
            faiss_path, _ = get_faiss_filenames(f)
            if os.path.exists(faiss_path):
                files_with_faiss.append(f)
        
        if not files_with_faiss:
            st.warning("No FAISS indexes found. Please process PDFs with embeddings enabled.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_file = st.selectbox(
                    "Select PDF to search in:",
                    options=["All PDFs"] + files_with_faiss
                )
            
            with col2:
                st.metric("Files with FAISS", len(files_with_faiss))
            
            top_k = st.slider("Number of results:", 1, 20, 10)
            semantic_query = st.text_input("Enter your semantic search query:", key="semantic_search")
            
            if semantic_query and st.button("Search with FAISS"):
                try:
                    # Initialize embedding model
                    embeddings_model = OllamaEmbeddings(model=embedding_model)
                    
                    # Get query embedding
                    with st.spinner("Generating query embedding..."):
                        query_embedding = embeddings_model.embed_query(semantic_query)
                    
                    # Determine which files to search
                    if selected_file == "All PDFs":
                        search_files = files_with_faiss
                    else:
                        search_files = [selected_file]
                    
                    all_results = []
                    
                    # Search in each selected PDF's FAISS index
                    with st.spinner("Searching FAISS indexes..."):
                        for pdf_file in search_files:
                            # Load FAISS index and metadata
                            index, metadata = load_pdf_faiss_index(pdf_file)
                            
                            if index is None:
                                continue
                            
                            # Search FAISS index
                            results = search_faiss_index(index, metadata, query_embedding, top_k)
                            
                            # Get corresponding Q&A pairs
                            for similarity, qa_idx in results:
                                if qa_idx < len(df):
                                    row = df.iloc[qa_idx]
                                    all_results.append({
                                        'file': row['file'],
                                        'similarity': similarity,
                                        'question': row['question'],
                                        'answer': row['answer']
                                    })
                    
                    # Sort by similarity and get top results
                    all_results.sort(key=lambda x: x['similarity'], reverse=True)
                    top_results = all_results[:top_k]
                    
                    st.success(f"🎯 Found {len(top_results)} results using FAISS")
                    
                    for idx, result in enumerate(top_results, 1):
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**#{idx} - File:** `{result['file']}`")
                            with col2:
                                st.markdown(f"**Similarity:** `{result['similarity']:.3f}`")
                            
                            st.markdown(f"**Q:** {result['question']}")
                            st.markdown(f"**A:** {result['answer']}")
                            st.divider()
                
                except Exception as e:
                    st.error(f"Error during FAISS search: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Download existing dataset
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Existing Dataset",
        data=csv_data,
        file_name=OUTPUT_CSV,
        mime="text/csv",
        key="download_existing"
    )
    
    # Option to clear dataset
    st.divider()
    st.subheader("🗑️ Cleanup Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clear Q&A Dataset", type="secondary"):
            if os.path.exists(OUTPUT_CSV):
                os.remove(OUTPUT_CSV)
                st.success("Q&A dataset cleared!")
                st.rerun()
    
    with col2:
        if st.button("Clear All FAISS Indexes", type="secondary"):
            if os.path.exists(EMBEDDINGS_DIRECTORY):
                faiss_files = [f for f in os.listdir(EMBEDDINGS_DIRECTORY) 
                              if f.endswith('.faiss') or f.endswith('.pkl')]
                for file in faiss_files:
                    os.remove(os.path.join(EMBEDDINGS_DIRECTORY, file))
                st.success(f"Deleted {len(faiss_files)} FAISS-related files!")
                st.rerun()
    
    with col3:
        if st.button("Clear Everything", type="secondary"):
            if os.path.exists(OUTPUT_CSV):
                os.remove(OUTPUT_CSV)
            if os.path.exists(EMBEDDINGS_DIRECTORY):
                all_files = [f for f in os.listdir(EMBEDDINGS_DIRECTORY) 
                           if f.endswith('.faiss') or f.endswith('.pkl')]
                for file in all_files:
                    os.remove(os.path.join(EMBEDDINGS_DIRECTORY, file))
            st.success("All data cleared!")
            st.rerun()

# --- FAISS Index Management Section ---
if os.path.exists(EMBEDDINGS_DIRECTORY):
    faiss_files = [f for f in os.listdir(EMBEDDINGS_DIRECTORY) if f.endswith('.faiss')]
    
    if faiss_files:
        st.divider()
        st.header("🗂️ FAISS Index Management")
        
        st.write(f"Found {len(faiss_files)} FAISS indexes:")
        
        for faiss_file in faiss_files:
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.text(f"📁 {faiss_file}")
            
            with col2:
                # Show file size
                file_path = os.path.join(EMBEDDINGS_DIRECTORY, faiss_file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                st.caption(f"{file_size:.1f} KB")
            
            with col3:
                # Show vector count
                try:
                    index = faiss.read_index(file_path)
                    st.caption(f"{index.ntotal} vectors")
                except:
                    st.caption("Error")
            
            with col4:
                # Delete individual FAISS index
                if st.button("🗑️", key=f"delete_{faiss_file}"):
                    # Delete both .faiss and .pkl files
                    base_name = faiss_file.replace('.faiss', '')
                    faiss_path = file_path
                    metadata_path = os.path.join(EMBEDDINGS_DIRECTORY, f"{base_name}_metadata.pkl")
                    
                    if os.path.exists(faiss_path):
                        os.remove(faiss_path)
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)
                    
                    st.success(f"Deleted {faiss_file} and metadata")
                    st.rerun()

st.divider()
st.caption("💡 Tip: This tool uses semantic similarity to find related chunks, creating more contextually-aware Q&A pairs!")