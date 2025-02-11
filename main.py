#!/usr/bin/env python
# coding: utf-8

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from redis import Redis

# Langchain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# ---------------------
# Setup Logging
# ---------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------
# Load Environment Variables
# ---------------------
load_dotenv()
# For local development you can hard-code (or use .env) your API key:
os.environ["OPENAI_API_KEY"] = "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"

class Config:
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    CHROMA_PERSIST_DIR = "chroma_db"
    MODEL_NAME = "gpt-4-turbo-preview"
    TEMPERATURE = 0.7
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MEMORY_WINDOW_SIZE = 10
    MAX_TOKENS = 4000
    CACHE_EXPIRE = 3600

# ---------------------
# Knowledge Base and Backup Content
# ---------------------
KNOWLEDGE_BASE = {
    "platform": {
        "url": "https://artisan.co/platform",
        "patterns": {
            "sales_platform": "div[data-section='artisan-sales-platform']",
            "ai_agent": "div[data-section='ava-ai-agent']"
        }
    },
    "products": {
        "url": "https://artisan.co/products",
        "patterns": {
            "linkedin_outreach": "div[data-section='linkedin-outreach']",
            "email_warmup": "div[data-section='email-warmup']",
            "sales_automation": "div[data-section='sales-automation']",
            "email_personalization": "div[data-section='email-personalization']",
            "deliverability": "div[data-section='deliverability']"
        }
    },
    "data": {
        "url": "https://artisan.co/data",
        "patterns": {
            "b2b_data": "div[data-section='b2b-data']",
            "ecommerce_data": "div[data-section='ecommerce-data']",
            "local_data": "div[data-section='local-data']"
        }
    }
}

BACKUP_CONTENT = {
    "platform": {
        "sales_platform": """
        Artisan Sales Platform features:
        - Unified sales workflow
        - Integrated communication tools
        - Analytics and reporting
        - Team collaboration features
        """,
        "ai_agent": """
        Ava AI Agent capabilities:
        - Automated lead engagement
        - Smart response generation
        - Conversation analysis
        - Follow-up scheduling
        """
    },
    "products": {
        "email_warmup": """
        Email Warmup service includes:
        - Gradual volume increase
        - Engagement automation
        - Provider optimization
        - Reputation monitoring
        """
    }
}

# ---------------------
# Pydantic Models
# ---------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str = Field(..., min_length=1, max_length=50)
    context: Optional[Dict[str, Any]] = None

    @validator('message')
    def clean_message(cls, v):
        return v.strip()

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_actions: List[str] = []
    related_topics: List[str] = []

# ---------------------
# Redis Memory Manager
# ---------------------
class RedisConversationMemory:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.max_history = 10

    def save_message(self, conversation_id: str, role: str, content: str):
        key = f"chat:{conversation_id}"
        message = json.dumps({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        pipeline = self.redis.pipeline()
        pipeline.lpush(key, message)
        pipeline.ltrim(key, 0, self.max_history - 1)
        pipeline.execute()

    def get_history(self, conversation_id: str) -> List[Dict]:
        key = f"chat:{conversation_id}"
        messages = self.redis.lrange(key, 0, self.max_history - 1)
        return [json.loads(msg) for msg in messages][::-1]

    def format_history_for_prompt(self, conversation_id: str) -> str:
        history = self.get_history(conversation_id)
        return "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in history
        ])

# ---------------------
# Knowledge Manager with Scraping
# ---------------------
class KnowledgeManager:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
        self.vectorstore = self._initialize_vectorstore()
        self.last_update = None
        self.update_interval = timedelta(hours=24)
        self._load_backup_content()

    def _load_backup_content(self):
        self.backup_content = BACKUP_CONTENT

    async def _fetch_content(self, url: str, patterns: Dict[str, str]) -> Dict[str, str]:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        return self._parse_content(html, patterns)
                    else:
                        logger.warning(f"Failed to fetch {url}: {response.status}")
                        return {}
            except Exception as e:
                logger.error(f"Error fetching {url}: {str(e)}")
                return {}

    def _parse_content(self, html: str, patterns: Dict[str, str]) -> Dict[str, str]:
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup.find_all(['script', 'style', 'nav', 'footer']):
            tag.decompose()
        content = {}
        for key, selector in patterns.items():
            elements = soup.select(selector)
            if elements:
                content[key] = "\n".join(element.get_text(strip=True, separator=" ") for element in elements)
        return content

    def _initialize_vectorstore(self) -> Chroma:
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
            if os.path.exists(Config.CHROMA_PERSIST_DIR):
                logger.info("Loading existing vector store...")
                vectorstore = Chroma(
                    persist_directory=Config.CHROMA_PERSIST_DIR,
                    embedding_function=embeddings
                )
                return vectorstore
            return Chroma(embedding_function=embeddings)
        except Exception as e:
            logger.error(f"Vector store initialization failed: {str(e)}")
            raise

    async def update_knowledge_base(self):
        if self.last_update and datetime.now() - self.last_update < self.update_interval:
            logger.info("Knowledge base update not required yet.")
            return

        all_content = []
        for section, config in KNOWLEDGE_BASE.items():
            try:
                content = await self._fetch_content(config['url'], config['patterns'])
                if not content and section in self.backup_content:
                    content = self.backup_content[section]
                if content:
                    self.redis.hset(f"knowledge:{section}", mapping=content)
                    all_content.extend(content.values())
            except Exception as e:
                logger.error(f"Error updating {section}: {str(e)}")

        if all_content:
            await self._update_vectorstore(all_content)
        self.last_update = datetime.now()

    async def _update_vectorstore(self, content: List[str]):
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            texts = text_splitter.split_text("\n\n".join(content))
            vectorstore = Chroma.from_texts(
                texts,
                self.embeddings,
                persist_directory=Config.CHROMA_PERSIST_DIR
            )
            vectorstore.persist()
            self.vectorstore = vectorstore
        except Exception as e:
            logger.error(f"Vector store update failed: {str(e)}")

    def get_relevant_content(self, query: str) -> List[str]:
        try:
            docs = self.vectorstore.similarity_search(query, k=3)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Error retrieving content: {str(e)}")
            return []

# ---------------------
# Conversation Manager
# ---------------------
class ConversationManager:
    def __init__(self, knowledge_manager: KnowledgeManager, redis_client: Redis):
        self.knowledge_manager = knowledge_manager
        self.memory = RedisConversationMemory(redis_client)
        self.llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=Config.TEMPERATURE)
        self._setup_prompt_template()

    def _setup_prompt_template(self):
        self.prompt = PromptTemplate(
            template="""
You are an AI assistant for Artisan, specializing in:
1. Sales Platform and AI Agent (Ava)
2. Email Warmup and Deliverability
3. LinkedIn Outreach and Sales Automation
4. Data Services (B2B, E-commerce, Local)

Current context: {context}
Chat history: {chat_history}
Question: {question}

Provide a response that:
1. Directly addresses the question
2. References specific Artisan features
3. Includes relevant examples
4. Suggests next steps when appropriate

Response:""",
            input_variables=["context", "chat_history", "question"]
        )

    def get_conversation_chain(self, conversation_id: str) -> ConversationalRetrievalChain:
        history = self.memory.get_history(conversation_id)
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=Config.MEMORY_WINDOW_SIZE,
            output_key="answer"
        )
        
        # Load history into memory
        for msg in history:
            if msg["role"] == "user":
                memory.save_context({"input": msg["content"]}, {"answer": ""})
            else:
                memory.save_context({"input": ""}, {"answer": msg["content"]})
                
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.knowledge_manager.vectorstore.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            return_source_documents=True,
            verbose=True
        )

def _extract_related_topics(query: str, response: str) -> List[str]:
    topics = []
    for section in KNOWLEDGE_BASE.keys():
        if section.lower() in query.lower() or section.lower() in response.lower():
            topics.append(section)
    return topics[:3]

def _generate_suggested_actions(response: str) -> List[str]:
    actions = []
    if "setup" in response.lower():
        actions.append("Book a demo for setup guidance")
    if "email warmup" in response.lower():
        actions.append("Check email warmup dashboard")
    if "campaign" in response.lower():
        actions.append("Review campaign templates")
    return actions[:3]

# ---------------------
# FastAPI App Setup
# ---------------------
app = FastAPI(title="Artisan AI Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis and Managers
redis_client = Redis.from_url(Config.REDIS_URL, decode_responses=True)
knowledge_manager = KnowledgeManager(redis_client)
conversation_manager = ConversationManager(knowledge_manager, redis_client)

# ---------------------
# API Endpoints
# ---------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        # Save the user's message.
        conversation_manager.memory.save_message(request.conversation_id, "user", request.message)
        
        # Get response using the conversation chain.
        chain = conversation_manager.get_conversation_chain(request.conversation_id)
        result = chain({"question": request.message})
        
        # Save the assistant's response.
        conversation_manager.memory.save_message(request.conversation_id, "assistant", result["answer"])
        
        return ChatResponse(
            response=result["answer"],
            sources=[doc.page_content[:200] for doc in result.get("source_documents", [])],
            confidence=min(len(result.get("source_documents", [])) / 3.0, 1.0),
            suggested_actions=_generate_suggested_actions(result["answer"]),
            related_topics=_extract_related_topics(request.message, result["answer"])
        )
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_store": bool(knowledge_manager.vectorstore),
        "last_update": knowledge_manager.last_update,
        "redis_connected": redis_client.ping()
    }

@app.get("/")
async def root():
    return {"message": "Welcome to the Artisan AI Assistant API!"}

@app.get("/history/count/{conversation_id}")
async def get_history_count(conversation_id: str):
    history = conversation_manager.memory.get_history(conversation_id)
    return {"message_count": len(history)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
