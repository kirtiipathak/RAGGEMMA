import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import PromptTemplate
import chainlit as cl
from typing import List, Dict, Tuple, Any, Optional
from rapidfuzz import fuzz, process
import numpy as np
from langchain.schema import Document
from langchain_core.callbacks import Callbacks
from langchain.retrievers import ContextualCompressionRetriever, BM25Retriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

class HybridRetriever:
    """
    Custom retriever that combines vector similarity search with keyword search
    to get the best of both semantic and lexical matching.
    """
    
    def __init__(self, vectorstore, k=4, similarity_weight=0.7, keyword_weight=0.3):
        self.vectorstore = vectorstore
        self.k = k
        self.similarity_weight = similarity_weight
        self.keyword_weight = keyword_weight
        # Extract all documents for keyword search
        self.documents = self._get_all_documents()
    
    def _get_all_documents(self):
        """Get all documents from the vectorstore if possible"""
        if hasattr(self.vectorstore, 'get'):
            return self.vectorstore.get()
        elif hasattr(self.vectorstore, '_collection'):
            # For Chroma
            return [Document(page_content=doc.page_content, metadata=doc.metadata) 
                    for doc in self.vectorstore._collection.get()]
        else:
            print("Warning: Could not extract all documents from vectorstore")
            return []
    
    def _normalize_similarity_score(self, score):
        """Normalize similarity score to 0-1 range"""
        # Assuming cosine similarity (-1 to 1)
        return (score + 1) / 2
    
    def _calculate_keyword_score(self, query, document):
        """Calculate fuzzy match score between query and document"""
        content = document.page_content.lower()
        query = query.lower()
        
        # Calculate multiple fuzzy metrics and combine them
        token_ratio = fuzz.token_sort_ratio(query, content) / 100
        partial_ratio = fuzz.partial_ratio(query, content) / 100
        
        # Average the scores with weights
        return (token_ratio * 0.7) + (partial_ratio * 0.3)
    
    def get_relevant_documents(self, query: str, filter: Optional[Dict] = None) -> List[Document]:
        """
        Retrieve relevant documents using hybrid search approach.
        
        Args:
            query: The search query
            filter: Optional filter to apply to the search
            
        Returns:
            List of relevant documents
        """
        # Perform vector similarity search
        similarity_results = self.vectorstore.similarity_search_with_score(
            query, k=self.k * 2, filter=filter
        )
        
        # Normalize similarity scores
        normalized_similarity = [(doc, self._normalize_similarity_score(score) * self.similarity_weight) 
                                 for doc, score in similarity_results]
        
        # Perform keyword search
        keyword_results = []
        for doc in self.documents:
            if filter and not self._apply_filter(doc, filter):
                continue
                
            # Calculate keyword score
            score = self._calculate_keyword_score(query, doc) * self.keyword_weight
            keyword_results.append((doc, score))
        
        # Sort and get top k keyword results
        keyword_results.sort(key=lambda x: x[1], reverse=True)
        keyword_results = keyword_results[:self.k * 2]
        
        # Combine results
        all_results = normalized_similarity + keyword_results
        
        # Group by document ID to avoid duplicates, keeping highest score
        unique_results = {}
        for doc, score in all_results:
            doc_id = doc.metadata.get('source', id(doc))
            if doc_id not in unique_results or score > unique_results[doc_id][1]:
                unique_results[doc_id] = (doc, score)
        
        # Get final sorted results
        final_results = sorted(unique_results.values(), key=lambda x: x[1], reverse=True)[:self.k]
        
        # Print some debug info
        print(f"Hybrid search results for query: '{query}'")
        for doc, score in final_results:
            print(f"Score: {score:.4f} - {doc.page_content[:50]}...")
        
        # Return just the documents
        return [doc for doc, _ in final_results]
    
    def _apply_filter(self, doc: Document, filter: Dict) -> bool:
        """Apply filter to document"""
        for key, value in filter.items():
            if key not in doc.metadata or doc.metadata[key] != value:
                return False
        return True

    def as_retriever(self):
        """Return self as a retriever-like object"""
        return self

# Global LLM instance
llm = ChatOllama(model="phi3")

@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=100,
            timeout=180,
        ).send()

    file = files[0]
    
    # Inform the user that processing has started
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    try:
        # Read the PDF file
        pdf = PyPDF2.PdfReader(file.path)
        pdf_text = ""
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            pdf_text += f"\n\n[Page {page_num + 1}]\n{page_text}"

        print(f"Extracted text length: {len(pdf_text)}")

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        texts = text_splitter.split_text(pdf_text)

        print(f"Number of text chunks: {len(texts)}")

        # Create enhanced metadata for each chunk
        metadatas = []
        for i, chunk in enumerate(texts):
            # Identify which page this chunk likely belongs to
            page_marker = f"[Page "
            page_num = None
            
            for j in range(len(pdf.pages)):
                if f"[Page {j + 1}]" in chunk:
                    page_num = j + 1
                    break
            
            metadatas.append({
                "source": f"chunk_{i}",
                "chunk_id": i,
                "document": file.name,
                "page": page_num,
                "chunk_size": len(chunk)
            })

        # Create embeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Create a Chroma vector store
        docsearch = await cl.make_async(Chroma.from_texts)(
            texts, embeddings, metadatas=metadatas
        )
        
        # Create a BM25 retriever for lexical search
        bm25_retriever = BM25Retriever.from_texts(texts)
        
        # Set metadata for BM25 documents to match Chroma's
        for i, doc in enumerate(bm25_retriever.docs):
            doc.metadata = metadatas[i]
        
        # Create a hybrid retriever
        hybrid_retriever = HybridRetriever(
            vectorstore=docsearch,
            k=5,  # Retrieve 5 documents
            similarity_weight=0.7,  # Weight for vector similarity
            keyword_weight=0.3  # Weight for keyword matching
        )
        
        # Create an ensemble retriever that combines multiple retrievers
        ensemble_retriever = EnsembleRetriever(
            retrievers=[
                hybrid_retriever,
                bm25_retriever
            ],
            weights=[0.7, 0.3]
        )
        
        # Add a contextual compression layer to extract relevant parts
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
        
        # Initialize message history for conversation
        message_history = ChatMessageHistory()
        
        # Memory for conversational context
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )

        # Create a custom prompt template
        prompt_template = """
        You are answering questions about a document. Use the following pieces of context to answer the question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}
        
        Chat History:
        {chat_history}
        
        Question: {question}
        
        Answer the question thoroughly, citing specific parts of the document when possible.
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question", "chat_history"]
        )

        # Create a chain that uses the Chroma vector store
        chain = ConversationalRetrievalChain.from_llm(
            llm,
            chain_type="stuff",
            retriever=compression_retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )

        # Let the user know that the system is ready
        msg.content = f"Processing `{file.name}` done. You can now ask questions about the PDF! Using advanced retrieval methods."
        await msg.update()
        
        # Store the chain in user session
        cl.user_session.set("chain", chain)

    except Exception as e:
        error_msg = f"An error occurred while processing the PDF: {str(e)}"
        print(error_msg)
        await cl.Message(content=error_msg).send()


@cl.on_message
async def main(message: cl.Message):
    if message.content.lower() == "upload new pdf":
        await on_chat_start()
        return
        
    try:
        # Retrieve the chain from user session
        chain = cl.user_session.get("chain") 
        print(f"Received message: {message.content}")
        
        # Show thinking message
        thinking_msg = cl.Message(content="Searching document and analyzing...")
        await thinking_msg.send()
        
        # Call the chain with user's message content
        res = await chain.ainvoke(message.content)
        
        answer = res["answer"]
        source_documents = res["source_documents"] 
        
        # Process source documents
        text_elements = []
        source_citations = []
        
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            
            # Create a cleaner citation
            page_num = source_doc.metadata.get('page', 'unknown')
            chunk_id = source_doc.metadata.get('chunk_id', source_idx)
            citation = f"[{source_idx+1}]"
            
            if page_num is not None:
                citation += f" Page {page_num}"
            
            source_citations.append(citation)
            
            # Create text element with citation
            text_elements.append(
                cl.Text(content=f"{citation}: {source_doc.page_content}", name=source_name)
            )
        
        # Add citations to the answer
        if source_citations:
            answer += "\n\nSources: " + ", ".join(source_citations)
        
        # Generate a summary of key points
        if source_documents:
            combined_text = "\n\n".join([doc.page_content for doc in source_documents])
            summary_prompt = f"""
            Based on the following text, provide a concise summary of the key information relevant to the question: "{message.content}"
            
            TEXT:
            {combined_text}
            
            SUMMARY (3-5 bullet points):
            """
            
            summary_response = await llm.ainvoke(summary_prompt)
            summary = summary_response.content
            
            # Add summary to response
            enhanced_answer = f"{answer}\n\n**Key Points:**\n{summary}"
        else:
            enhanced_answer = answer
        
        # Update thinking message
        thinking_msg.content = enhanced_answer
        await thinking_msg.update()
        
        # Send sources as a separate message
        if text_elements:
            await cl.Message(content="**Source Documents:**", elements=text_elements).send()

    except Exception as e:
        error_msg = f"An error occurred while processing your question: {str(e)}"
        print(error_msg)
        if hasattr(e, 'response'):
            print(f"Response content: {e.response.content}")
        await cl.Message(content=error_msg).send()

# Add a new command handler for uploading a new PDF
@cl.on_message
async def handle_message(message: cl.Message):
    if message.content.lower() == "upload new pdf":
        await on_chat_start()
    else:
        await main(message)