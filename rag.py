# rag_pipeline.py

import os
import shutil
from typing import List, Optional, Tuple

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from prompts import *
from utils import extract_content, extract_content_list


class TranscriptIndexer:

    def __init__(
        self,
        embedding_model_name: str = "text-embedding-3-small",
        index_path: Optional[str] = None,
        collection_name: str = "transcript_chunks"
    ):
        """
        Args:
          embedding_model_name: OpenAI embedding model (e.g., "text-embedding-3-small").
          index_path: Directory where Chroma will persist embeddings/indices.
          collection_name: Name of the Chroma collection to create/load.
        """
        self.embeddings = OpenAIEmbeddings(model=embedding_model_name)
        self.index_path = index_path
        self.collection_name = collection_name
        self.retriever = None

    def save_index(self) -> None:
        """
        Persists the current Chroma collection to disk (if using a persist directory).
        """
        if self.retriever is not None:
            # The retriever wraps a vectorstore, so we grab its underlying vectorstore
            self.retriever.vectorstore.persist()
            
    def load_json(
        self,
        json_path: str
    ) -> List[dict]:
        """
        Reads a JSON file containing a list of utterances and returns it.
        Args:
            json_path: Path to the JSON file containing utterances.
        Returns:
            List of utterance dictionaries, each with keys like "text", "turn_index", "speaker".
        """
        import json
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("utterances", [])

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 300,
        chunk_overlap: int = 50
    ) -> List[str]:
        """
        Splits the input text into overlapping chunks.

        Args:
            text: The full transcript text to split.
            chunk_size: Maximum # of characters per chunk.
            chunk_overlap: Overlap (in characters) between consecutive chunks.

        Returns:
            List of chunk strings.
        """
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)

    def index_transcript(self, transcript: List[dict], rebuild: bool = False) -> None:
        """
        Given a raw transcript json, index it into the Chroma vector store.
        Args:
            transcript: List of utterance dictionaries, each with keys like "text", "turn_index", "speaker".
            rebuild: If True, will delete existing index and rebuild from scratch.
        Returns:
            None
        """
        if self.index_path and rebuild and os.path.isdir(self.index_path):
            shutil.rmtree(self.index_path)

        if self.index_path and os.path.isdir(self.index_path) and not rebuild:
            vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.index_path,
                collection_name=self.collection_name
            )
            self.retriever = vectorstore.as_retriever(
                search_type="similarity",
            )
            return

        vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.index_path,
            collection_name=self.collection_name
        )

        for utt in transcript:
            if not isinstance(utt, dict) or "text" not in utt:
                raise ValueError("Each utterance must be a dict with a 'text' key.")
            chunks = self.chunk_text(utt["text"])

            for chunk in chunks:
                vectorstore.add_texts(
                    texts=[chunk],
                    metadatas=[{
                        "turn_index": utt["turn_index"],
                        "speaker": utt["speaker"],
                    }]
                )
        # Save the index to disk if an index path is specified
        if self.index_path:
            vectorstore.persist()

        self.retriever = vectorstore.as_retriever(
            search_type="similarity"
        )

    def retrieve(
        self, query: str,
        top_k: int = 3,
        filter_by: Optional[dict] = None
    ) -> List[str]:
        """
        Given a text `query`, retrieve the top_k most relevant chunks from the Chroma index.
        Args:
            query: The text query to search for in the indexed transcript.
            top_k: How many chunks to return.
            filter_by: Optional metadata filter (e.g., {"speaker": "Clinician"}).

        Returns:
            A list of unique chunk strings. Results sorted by descending similarity.
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call `index_transcript(...)` first.")
        
        self.retriever.search_kwargs = {"k": top_k}
        docs = self.retriever.get_relevant_documents(query, filter=filter_by)
        
        # Extract page content and remove any duplicates
        chunks: List[str] = []
        seen = set()
        for doc in docs:
            text = doc.page_content
            if text not in seen:
                seen.add(text)
                chunks.append(text)
        return chunks


class SOAPGenerator:
    def __init__(
        self,
        indexer: TranscriptIndexer,
        input_path: Optional[str] = None,
        transcript: Optional[List[dict]] = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        topk: int = 4,
        rebuild: bool = False
    ):
        """
        Args:
          indexer: A TranscriptIndexer instance.
          model_name: OpenAI chat model.
          temperature: controls randomness in the model's responses (0.0 = deterministic).
          max_tokens: maximum tokens to allow in the assistant’s reply.
          topk: how many chunks to fetch for each section.
          rebuild: if True, will delete existing index and rebuild from scratch.
        """
        self.indexer = indexer
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.top_k = topk
        
        self.init_vectorstore(
            input_path=input_path,
            transcript=transcript,
            rebuild=rebuild
        )
        
    def init_vectorstore(
        self,
        input_path: Optional[str] = None,
        transcript: Optional[List[dict]] = None,
        rebuild: bool = False
    ) -> None:
        if input_path is None and transcript is None:
            raise ValueError("Must provide either input_path or transcript data.")
        if transcript is None:
            transcript = self.indexer.load_json(input_path)

        self.indexer.index_transcript(
            transcript,
            rebuild=rebuild
        )
        
    def prepare_prompt(
        self,
        prompt_template: str,
    ):
        return ChatPromptTemplate.from_template(prompt_template)

    def generate_soap(self,):
        """
        Generates a SOAP note from the indexed transcript.
        Returns:
            A formatted SOAP note string.
        """
        extract_prompt = self.prepare_prompt(EXTRACT_PROMPT)
        extract_chain = extract_prompt | self.llm | StrOutputParser()
        # Clinician name extraction
        clinician_name_query = self.indexer.retrieve(
            "What is the name of clinician.",
            top_k=20,
            filter_by={"speaker": "Clinician"}
        )
        clinician_name_query = "\n".join(clinician_name_query)
        clinician_name_response = extract_chain.invoke(
            {
                "context": clinician_name_query,
                "subject": "clinician name"
            }
        )
        clinician_name = extract_content_list(clinician_name_response)
        # Patient name extraction
        patient_name_query = self.indexer.retrieve(
            "What is the name of patient.",
            top_k=20,
            filter_by={"speaker": "Patient"}
        )
        patient_name_query = "\n".join(patient_name_query)
        patient_name_response = extract_chain.invoke(
            {
                "context": patient_name_query,
                "subject": "patient name"
            }
        )
        patient_name = extract_content_list(patient_name_response)
        # Date extraction
        date_query = self.indexer.retrieve(
            "What is the date of the encounter.",
            top_k=20
        )
        date_query = "\n".join(date_query)
        date_encounter_response = extract_chain.invoke(
            {
                "context": date_query,
                "subject": "date of encounter"
            }
        )
        date_encounter = extract_content_list(date_encounter_response)
        query_prompt = SUBJECTIVE_Q_PROMPT.format(
            system=Q_SYS_PROMPT
        )
        # Retrieve subjective information
        subjective_query = self.indexer.retrieve(query_prompt, top_k=self.top_k)
        subjective_query = "\n".join(subjective_query)
        subjective_write_prompt = self.prepare_prompt(SUBJECTIVE_W_PROMPT)
        subjective_write_chain = subjective_write_prompt | self.llm | StrOutputParser()
        subjective_content_response = subjective_write_chain.invoke(
            {"system": W_SYS_PROMPT, "context": subjective_query, "rules": W_RULE_PROMPT}
        )
        # Retrieve objective information
        objective_query_prompt = OBJECTIVE_Q_PROMPT.format(
            system=Q_SYS_PROMPT
        )
        objective_query = self.indexer.retrieve(objective_query_prompt, top_k=self.top_k)
        objective_query = "\n".join(objective_query)
        objective_write_prompt = self.prepare_prompt(OBJECTIVE_W_PROMPT)
        objective_write_chain = objective_write_prompt | self.llm | StrOutputParser()
        objective_content_response = objective_write_chain.invoke(
            {"system": W_SYS_PROMPT, "context": objective_query, "rules": W_RULE_PROMPT}
        )
        # Retrieve assessment information
        assessment_query_prompt = ASSESSMENT_Q_PROMPT.format(
            system=Q_SYS_PROMPT
        )
        assessment_query = self.indexer.retrieve(assessment_query_prompt, top_k=self.top_k)
        assessment_query = "\n".join(assessment_query)
        assessment_write_prompt = self.prepare_prompt(ASSESSMENT_W_PROMPT)
        assessment_write_chain = assessment_write_prompt | self.llm | StrOutputParser()
        assessment_content_response = assessment_write_chain.invoke(
            {"system": W_SYS_PROMPT, "context": assessment_query, "rules": W_RULE_PROMPT}
        )
        # Retrieve plan information
        plan_query_prompt = PLAN_Q_PROMPT.format(
            system=Q_SYS_PROMPT
        )
        plan_query = self.indexer.retrieve(plan_query_prompt, top_k=self.top_k)
        plan_query = "\n".join(plan_query)
        plan_write_prompt = self.prepare_prompt(PLAN_W_PROMPT)
        plan_write_chain = plan_write_prompt | self.llm | StrOutputParser()
        plan_content_response = plan_write_chain.invoke(
            {"system": W_SYS_PROMPT, "context": plan_query, "rules": W_RULE_PROMPT}
        )
        
        subjective_content = extract_content(subjective_content_response)
        objective_content = extract_content(objective_content_response)
        assessment_content = extract_content(assessment_content_response)
        plan_content = extract_content(plan_content_response)
        clinician_name = ",".join(clinician_name) if clinician_name else "N/A"
        patient_name = ",".join(patient_name) if patient_name else "N/A"
        date_encounter = ",".join(date_encounter) if date_encounter else "N/A"
        
        soap_content = SOAP_NOTE_PROMPT.format(
            clinician=clinician_name,
            patient=patient_name,
            date=date_encounter,
            subjective=subjective_content,
            objective=objective_content,
            assessment=assessment_content,
            plan=plan_content
        )
        return soap_content

