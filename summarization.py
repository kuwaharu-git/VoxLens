"""
Summarization module using LangChain and Ollama
"""
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.docstore.document import Document
import config


class ConversationSummarizer:
    """Summarize conversations using LangChain and Ollama"""
    
    def __init__(self, model_name: str = None, base_url: str = None):
        """
        Initialize the summarizer
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name or config.LLM_MODEL
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.llm = None
        
    def _initialize_llm(self):
        """Initialize the Ollama LLM"""
        if self.llm is None:
            self.llm = Ollama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=0.3
            )
    
    def summarize(self, transcription: str, use_map_reduce: bool = False) -> str:
        """
        Summarize the transcription considering speaker relationships
        
        Args:
            transcription: Full transcription with speaker labels
            use_map_reduce: Whether to use MapReduce for long documents
            
        Returns:
            Summary text
        """
        self._initialize_llm()
        
        # Create document from transcription
        doc = Document(page_content=transcription)
        
        # Define prompt template for summarization
        prompt_template = """以下は話者ごとに分類された会話の文字起こしです。
話者間の関係性や会話の流れを考慮して、重要なポイントを抽出し、簡潔な要約を作成してください。

文字起こし:
{text}

要約:"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Check if we should use MapReduce for long documents
        if use_map_reduce or len(transcription) > 4000:
            # Use MapReduce for long documents
            chain = load_summarize_chain(
                self.llm,
                chain_type="map_reduce",
                return_intermediate_steps=False
            )
            summary = chain.run([doc])
        else:
            # Use Stuff chain for shorter documents
            llm_chain = LLMChain(llm=self.llm, prompt=prompt)
            stuff_chain = StuffDocumentsChain(
                llm_chain=llm_chain,
                document_variable_name="text"
            )
            summary = stuff_chain.run([doc])
        
        return summary.strip()
    
    def summarize_with_custom_prompt(
        self, 
        transcription: str, 
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        Summarize with a custom prompt template
        
        Args:
            transcription: Full transcription with speaker labels
            custom_prompt: Custom prompt template (must include {text} placeholder)
            
        Returns:
            Summary text
        """
        self._initialize_llm()
        
        doc = Document(page_content=transcription)
        
        if custom_prompt:
            prompt = PromptTemplate.from_template(custom_prompt)
        else:
            # Use default prompt
            prompt_template = """以下の会話を要約してください:

{text}

要約:"""
            prompt = PromptTemplate.from_template(prompt_template)
        
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="text"
        )
        
        summary = stuff_chain.run([doc])
        return summary.strip()
