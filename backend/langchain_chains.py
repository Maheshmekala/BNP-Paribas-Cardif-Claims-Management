"""
LangChain integration for BNP Paribas Cardif Claims Management.

Provides:
- LLM chains for claim summarization
- Chain for fraud detection prompting
- Chain for document comparison
- ConversationBufferMemory for adjuster chat
"""

import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from backend.config import settings, logger


# ---------------------------------------------------------------------------
# Abstract LLM Provider
# ---------------------------------------------------------------------------
class LLMProvider:
    """
    Abstract LLM provider that tries real LLMs first, falls back to template-based responses.
    """

    def __init__(self):
        self._llm = None
        self._initialize()

    def _initialize(self):
        """Try to initialize a real LLM provider."""
        # Try OpenAI
        if settings.OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    model="gpt-3.5-turbo",
                    temperature=0.3,
                )
                logger.info("OpenAI LLM initialized.")
                return
            except ImportError:
                logger.warning("langchain-openai not installed.")
            except Exception as e:
                logger.warning(f"OpenAI init failed: {e}")

        # Try Anthropic
        if settings.ANTHROPIC_API_KEY:
            try:
                from langchain_anthropic import ChatAnthropic
                self._llm = ChatAnthropic(
                    api_key=settings.ANTHROPIC_API_KEY,
                    model="claude-3-haiku-20240307",
                    temperature=0.3,
                )
                logger.info("Anthropic LLM initialized.")
                return
            except ImportError:
                logger.warning("langchain-anthropic not installed.")
            except Exception as e:
                logger.warning(f"Anthropic init failed: {e}")

        logger.info("No LLM provider configured. Using template-based fallback.")

    def is_available(self) -> bool:
        """Check if a real LLM is available."""
        return self._llm is not None

    def get_llm(self):
        """Get the LLM instance."""
        return self._llm


llm_provider = LLMProvider()


# ---------------------------------------------------------------------------
# In-Memory Chat History Store
# ---------------------------------------------------------------------------
class ChatMemoryStore:
    """
    Stores conversation memories for adjuster chat sessions.
    Uses ConversationBufferMemory from LangChain.
    """

    def __init__(self):
        self._memories: Dict[str, Any] = {}

    def get_memory(self, session_id: str):
        """Get or create a conversation memory for the session."""
        if session_id not in self._memories:
            try:
                from langchain.memory import ConversationBufferMemory
                self._memories[session_id] = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                )
            except ImportError:
                self._memories[session_id] = SimpleMemory()
        return self._memories[session_id]

    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        if session_id in self._memories:
            self._memories[session_id].clear()


class SimpleMemory:
    """Fallback simple memory when LangChain isn't available."""

    def __init__(self):
        self.chat_history = []

    def clear(self):
        self.chat_history = []

    def save_context(self, inputs, outputs):
        self.chat_history.append({"input": inputs.get("input", ""), "output": outputs.get("output", "")})

    def load_memory_variables(self, inputs=None):
        return {"chat_history": self.chat_history}


memory_store = ChatMemoryStore()


# ---------------------------------------------------------------------------
# LangChain Chains
# ---------------------------------------------------------------------------
class ClaimChains:
    """
    Collection of LangChain chains for claim processing.
    Each chain uses the LLM provider if available, otherwise template-based logic.
    """

    @staticmethod
    def summarize_claim(claim_data: Dict[str, Any]) -> str:
        """
        Generate a concise summary of a claim.

        Args:
            claim_data: Dictionary with claim fields (claim_number, category,
                       policyholder, description, amount, status, etc.)

        Returns:
            Summarized text.
        """
        if llm_provider.is_available():
            return ClaimChains._llm_summarize_claim(claim_data)
        return ClaimChains._template_summarize_claim(claim_data)

    @staticmethod
    def _llm_summarize_claim(claim_data: Dict[str, Any]) -> str:
        """Use LLM to summarize a claim."""
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert insurance claims analyst at BNP Paribas Cardif. "
                           "Summarize the following claim concisely in 2-3 sentences."),
                ("human", "Claim Number: {claim_number}\n"
                          "Policyholder: {policyholder}\n"
                          "Category: {category}\n"
                          "Status: {status}\n"
                          "Amount Claimed: EUR {amount}\n"
                          "Description: {description}\n"
                          "Fraud Score: {fraud_score}"),
            ])

            chain = prompt | llm_provider.get_llm() | StrOutputParser()
            return chain.invoke({
                "claim_number": claim_data.get("claim_number", "N/A"),
                "policyholder": claim_data.get("policyholder_name", "N/A"),
                "category": claim_data.get("category", "N/A"),
                "status": claim_data.get("status", "N/A"),
                "amount": claim_data.get("amount_claimed", 0),
                "description": claim_data.get("description", "N/A")[:500],
                "fraud_score": claim_data.get("fraud_score", 0),
            })
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            return ClaimChains._template_summarize_claim(claim_data)

    @staticmethod
    def _template_summarize_claim(claim_data: Dict[str, Any]) -> str:
        """Template-based claim summarization."""
        cn = claim_data.get("claim_number", "N/A")
        ph = claim_data.get("policyholder_name", "N/A")
        cat = claim_data.get("category", "N/A")
        st = claim_data.get("status", "N/A")
        amt = claim_data.get("amount_claimed", 0)
        desc = claim_data.get("description", "No description provided")[:200]
        fs = claim_data.get("fraud_score", 0)

        return (
            f"Claim {cn}: {ph} filed a {cat} insurance claim (Status: {st}). "
            f"Amount claimed: EUR {amt:,.2f}. "
            f"Description: {desc}. "
            f"Fraud risk assessment: {'HIGH' if fs > 0.7 else 'MEDIUM' if fs > 0.3 else 'LOW'} (score: {fs:.2f})."
        )

    @staticmethod
    def analyze_fraud(claim_data: Dict[str, Any], indicators: List[str]) -> str:
        """
        Generate fraud analysis text based on claim data and indicators.

        Args:
            claim_data: Claim information dict.
            indicators: List of fraud indicator descriptions.

        Returns:
            Fraud analysis text.
        """
        if llm_provider.is_available():
            return ClaimChains._llm_analyze_fraud(claim_data, indicators)
        return ClaimChains._template_analyze_fraud(claim_data, indicators)

    @staticmethod
    def _llm_analyze_fraud(claim_data: Dict[str, Any], indicators: List[str]) -> str:
        """Use LLM for fraud analysis."""
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            indicators_text = "\n".join([f"- {ind}" for ind in indicators]) if indicators else "No specific indicators."

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a fraud detection specialist at BNP Paribas Cardif. "
                           "Analyze the fraud risk for this insurance claim based on the indicators provided. "
                           "Provide a concise risk assessment."),
                ("human", "Claim: {claim_number}\n"
                          "Category: {category}\n"
                          "Amount: EUR {amount}\n"
                          "Fraud Score: {score}\n\n"
                          "Indicators:\n{indicators}\n\n"
                          "Risk Assessment:"),
            ])

            chain = prompt | llm_provider.get_llm() | StrOutputParser()
            return chain.invoke({
                "claim_number": claim_data.get("claim_number", "N/A"),
                "category": claim_data.get("category", "N/A"),
                "amount": claim_data.get("amount_claimed", 0),
                "score": claim_data.get("fraud_score", 0),
                "indicators": indicators_text,
            })
        except Exception as e:
            logger.warning(f"LLM fraud analysis failed: {e}")
            return ClaimChains._template_analyze_fraud(claim_data, indicators)

    @staticmethod
    def _template_analyze_fraud(claim_data: Dict[str, Any], indicators: List[str]) -> str:
        """Template-based fraud analysis."""
        score = claim_data.get("fraud_score", 0)
        cn = claim_data.get("claim_number", "N/A")

        if score > 0.7:
            level = "HIGH"
            action = "Immediate manual review required. Recommend escalation to fraud investigation team."
        elif score > 0.3:
            level = "MEDIUM"
            action = "Additional verification recommended. Consider requesting supplementary documentation."
        else:
            level = "LOW"
            action = "Standard processing. No unusual patterns detected."

        ind_text = "\n".join([f"  - {i}" for i in indicators]) if indicators else "  - None"

        return (
            f"Fraud Risk Assessment for Claim {cn}:\n"
            f"Risk Level: {level} (Score: {score:.3f})\n\n"
            f"Indicators Found:\n{ind_text}\n\n"
            f"Action: {action}"
        )

    @staticmethod
    def compare_documents(doc_texts: List[str]) -> str:
        """
        Compare multiple document texts for consistency.

        Args:
            doc_texts: List of document text contents.

        Returns:
            Comparison analysis.
        """
        if len(doc_texts) < 2:
            return "Need at least 2 documents to compare."

        if llm_provider.is_available():
            return ClaimChains._llm_compare_documents(doc_texts)
        return ClaimChains._template_compare_documents(doc_texts)

    @staticmethod
    def _llm_compare_documents(doc_texts: List[str]) -> str:
        """Use LLM for document comparison."""
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            docs = "\n\n---DOCUMENT SEPARATOR---\n\n".join(
                [f"Document {i+1}:\n{t[:1000]}" for i, t in enumerate(doc_texts)]
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a document analysis expert at BNP Paribas Cardif. "
                           "Compare the following claim documents for consistency. "
                           "Identify any discrepancies, contradictions, or matching information."),
                ("human", "Documents for comparison:\n\n{docs}\n\n"
                          "Comparison Analysis:"),
            ])

            chain = prompt | llm_provider.get_llm() | StrOutputParser()
            return chain.invoke({"docs": docs})
        except Exception as e:
            logger.warning(f"LLM document comparison failed: {e}")
            return ClaimChains._template_compare_documents(doc_texts)

    @staticmethod
    def _template_compare_documents(doc_texts: List[str]) -> str:
        """Template-based document comparison."""
        analysis = "Document Comparison Analysis:\n\n"

        for i, text in enumerate(doc_texts):
            analysis += f"Document {i+1}: {len(text)} characters.\n"
            # Extract key fields
            import re

            # Look for dates
            dates = re.findall(r"\d{2}[/-]\d{2}[/-]\d{4}", text)
            if dates:
                analysis += f"  Dates found: {', '.join(dates)}\n"

            # Look for amounts
            amounts = re.findall(r"EUR\s*[\d,]+\.?\d*|€\s*[\d,]+\.?\d*", text)
            if amounts:
                analysis += f"  Amounts found: {', '.join(amounts)}\n"

            analysis += "\n"

        # Cross-document comparison
        if all("rear" in t.lower() for t in doc_texts):
            analysis += "Consistency: All documents reference rear-end collision.\n"
        if any("police" in t.lower() for t in doc_texts) and not all("police" in t.lower() for t in doc_texts):
            analysis += "DISCREPANCY: Police report mentioned in some documents but not all.\n"

        return analysis

    @staticmethod
    def chat_response(session_id: str, user_message: str) -> str:
        """
        Generate a chat response for the claim adjuster assistant.

        Uses ConversationBufferMemory for context.

        Args:
            session_id: Chat session identifier.
            user_message: User's message.

        Returns:
            Assistant response.
        """
        memory = memory_store.get_memory(session_id)

        if llm_provider.is_available():
            return ClaimChains._llm_chat_response(memory, user_message)

        return ClaimChains._template_chat_response(memory, user_message)

    @staticmethod
    def _llm_chat_response(memory, user_message: str) -> str:
        """Use LLM for chat response."""
        try:
            from langchain.chains import ConversationChain

            chain = ConversationChain(
                llm=llm_provider.get_llm(),
                memory=memory,
                verbose=False,
            )
            response = chain.predict(input=user_message)
            return response
        except Exception as e:
            logger.warning(f"LLM chat failed: {e}")
            return ClaimChains._template_chat_response(None, user_message)

    @staticmethod
    def _template_chat_response(memory, user_message: str) -> str:
        """Template-based chat response."""
        user_msg_lower = user_message.lower()

        # Save to memory if available
        if memory:
            memory.save_context({"input": user_message}, {"output": ""})

        # Pattern matching for common queries
        if "fraud" in user_msg_lower:
            response = (
                "Regarding fraud detection: Our system analyzes multiple indicators including "
                "claim timing, amount patterns, historical data, and document consistency. "
                "A fraud score above 0.7 triggers mandatory manual review. "
                "Key indicators include delayed filing, inconsistent descriptions, and high claim-to-coverage ratios."
            )
        elif "status" in user_msg_lower or "claim" in user_msg_lower:
            response = (
                "To check a claim's status, I need the claim number. You can also view all claims "
                "in the Claims List page. Claims go through: Submitted > In Review > "
                "Document Analysis > Fraud Check > Adjudication > Resolution."
            )
        elif "document" in user_msg_lower or "upload" in user_msg_lower:
            response = (
                "Documents can be uploaded through the Document Upload page. "
                "We support PDF, JPG, PNG, and CSV formats. "
                "Our system automatically performs OCR and extracts key information. "
                "For best results, ensure documents are clear and well-lit."
            )
        elif "rag" in user_msg_lower or "search" in user_msg_lower or "similar" in user_msg_lower:
            response = (
                "You can use the RAG Query page to search for similar claims semantically. "
                "Just describe the claim scenario and our system will find the most relevant "
                "historical claims from our vector database."
            )
        elif "dashboard" in user_msg_lower or "stats" in user_msg_lower or "analytics" in user_msg_lower:
            response = (
                "The Dashboard provides a comprehensive overview with charts showing "
                "claim status distribution, fraud trends, processing times, and monthly volumes. "
                "All charts are interactive and can be filtered."
            )
        elif "hello" in user_msg_lower or "hi" in user_msg_lower or "hey" in user_msg_lower:
            response = (
                "Hello! I'm your BNP Paribas Cardif Claims Assistant. "
                "I can help you with claim status inquiries, fraud analysis, document processing, "
                "and navigating the system. How can I assist you today?"
            )
        elif "help" in user_msg_lower:
            response = (
                "I can help you with:\n"
                "- Checking claim statuses and details\n"
                "- Fraud risk assessment explanations\n"
                "- Document upload guidance\n"
                "- RAG-based semantic search for similar claims\n"
                "- Dashboard and analytics explanations\n\n"
                "What would you like to know?"
            )
        else:
            response = (
                f"I understand you're asking about: '{user_message}'. "
                f"As a claims assistant, I can help with claims, documents, fraud analysis, and the "
                f"RAG system. Could you please be more specific about what you need?"
            )

        # Update memory with the response
        if memory:
            memory.save_context({"input": user_message}, {"output": response})

        return response


# Module-level singleton
claim_chains = ClaimChains()
