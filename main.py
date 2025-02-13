#!/usr/bin/env python
# coding: utf-8

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from starlette.requests import Request
from fastapi.responses import Response
import aiohttp
import psutil
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator, constr
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from redis import Redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Langchain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)


load_dotenv()


class Config:
    """Application configuration"""
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    API_KEY = os.environ.get("API_KEY")
    if not API_KEY:
        raise ValueError("API_KEY environment variable is required")

    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    CHROMA_PERSIST_DIR = "chroma_db"
    MODEL_NAME = "gpt-4-turbo-preview"
    TEMPERATURE = 0.7
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MEMORY_WINDOW_SIZE = 10
    MAX_TOKENS = 4000
    REDIS_KEY_TTL = 86400  # 24 hours
    REQUEST_TIMEOUT = 30  # seconds
    RATE_LIMIT = "100/hour"
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds_new

    # Vector store cleanup configuration
    VECTOR_STORE_MAX_AGE = timedelta(days=7)
    VECTOR_STORE_CLEANUP_INTERVAL = timedelta(days=1)

    # Allowed origins for CORS
    ALLOWED_ORIGINS = [
        "https://artisan.co",
        "https://app.artisan.co"
    ]

    @classmethod
    def get_cors_origins(cls):
        if os.getenv("ENVIRONMENT") == "development":
            return ["*"]
        return cls.ALLOWED_ORIGINS


api_key_header = APIKeyHeader(name="X-API-Key")


async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != Config.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return api_key


limiter = Limiter(key_func=get_remote_address)


KNOWLEDGE_BASE = {
    "sales_ai": {
        "url": "https://www.artisan.co/sales-ai",
        "patterns": {
            "features": "div.features",
            "benefits": "div.benefits",
            "use_cases": "div.use-cases"
        }
    },
    "ai_agent": {
        "url": "https://www.artisan.co/ai-sales-agent",
        "patterns": {
            "capabilities": "div.capabilities",
            "integrations": "div.integrations",
            "workflow": "div.workflow"
        }
    }
}

BACKUP_CONTENT = {
    "sales_ai": {
        "features": """
        - AI-powered sales automation
        - Intelligent lead scoring
        - Automated follow-ups
        - Performance analytics
        """,
        "benefits": """
        - Increased sales efficiency
        - Higher conversion rates
        - Reduced manual work
        - Better lead quality
        """,
        "use_cases": """
        - B2B lead generation
        - Sales pipeline automation
        - Customer engagement
        - Sales team productivity
        """
    },
    "ai_agent": {
        "capabilities": """
        - Natural language processing
        - Smart response generation
        - Context awareness
        - Multi-channel support
        """,
        "integrations": """
        - CRM systems
        - Email platforms
        - Communication tools
        - Analytics platforms
        """,
        "workflow": """
        - Lead qualification
        - Automated outreach
        - Follow-up scheduling
        - Performance tracking
        """
    }
}



class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str = Field(
        ..., 
        min_length=1, 
        max_length=50, 
        pattern="^[a-zA-Z0-9_-]+$"  # Using pattern instead of regex
    )
    context: Optional[Dict[str, Any]] = None

    @validator('message')
    def clean_message(cls, v):
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "message": "Tell me about email warmup service",
                "conversation_id": "user123_session456",
                "context": {"user_type": "admin"}
            }
        }


class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_actions: List[str] = []
    related_topics: List[str] = []


class HealthCheckResponse(BaseModel):
    status: str
    vector_store: bool
    vector_store_size: int
    last_update: Optional[datetime]
    redis_connected: bool
    memory_usage: Dict[str, int]
    uptime: float


class MetricsCollector:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    async def record_request(self, conversation_id: str, latency: float):
        timestamp = datetime.now().strftime("%Y-%m-%d:%H")
        self.redis.incr(f"metrics:requests:{timestamp}")
        self.redis.lpush(f"metrics:latency:{timestamp}", latency)
        self.redis.ltrim(f"metrics:latency:{timestamp}", 0, 999)
        self.redis.expire(f"metrics:latency:{timestamp}", 86400)  # 24 hours TTL

    def get_metrics(self) -> Dict[str, Any]:
        current_hour = datetime.now().strftime("%Y-%m-%d:%H")
        latencies = self.redis.lrange(f"metrics:latency:{current_hour}", 0, -1)
        avg_latency = sum(float(x) for x in latencies) / len(latencies) if latencies else 0

        return {
            "requests_current_hour": int(self.redis.get(f"metrics:requests:{current_hour}") or 0),
            "avg_latency": avg_latency,
            "total_conversations": len(self.redis.keys("chat:*"))
        }


class RedisConversationMemory:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.max_history = Config.MEMORY_WINDOW_SIZE

    async def save_message(self, conversation_id: str, role: str, content: str):
        key = f"chat:{conversation_id}"
        message = json.dumps({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        pipeline = self.redis.pipeline()
        pipeline.lpush(key, message)
        pipeline.ltrim(key, 0, self.max_history - 1)
        pipeline.expire(key, Config.REDIS_KEY_TTL)
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

    async def cleanup_old_conversations(self):
        """Cleanup conversations older than TTL"""
        for key in self.redis.scan_iter("chat:*"):
            if not self.redis.ttl(key):
                self.redis.delete(key)


class KnowledgeManager:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = self._initialize_vectorstore()
        self.last_update = None
        self.update_interval = timedelta(hours=24)
        self._load_backup_content()
        asyncio.create_task(self._cleanup_vector_store())

    def _load_backup_content(self):
        self.backup_content = BACKUP_CONTENT

    async def _fetch_content(self, url: str, patterns: Dict[str, str]) -> Dict[str, str]:
        for attempt in range(Config.MAX_RETRIES):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=Config.REQUEST_TIMEOUT) as response:
                        if response.status == 200:
                            html = await response.text()
                            return self._parse_content(html, patterns)
                        else:
                            logger.warning(f"Failed to fetch {url}: {response.status}")
            except Exception as e:
                logger.error(f"Error fetching {url} (attempt {attempt + 1}): {str(e)}")
                if attempt < Config.MAX_RETRIES - 1:
                    await asyncio.sleep(Config.RETRY_DELAY * (attempt + 1))
                continue
        return {}

    async def _cleanup_vector_store(self):
        while True:
            try:
                if os.path.exists(Config.CHROMA_PERSIST_DIR):
                    cutoff_date = datetime.now() - Config.VECTOR_STORE_MAX_AGE
                    # Implementation depends on your Chroma version and setup
                    logger.info(f"Cleaning up vector store entries older than {cutoff_date}")

                await asyncio.sleep(Config.VECTOR_STORE_CLEANUP_INTERVAL.total_seconds())
            except Exception as e:
                logger.error(f"Vector store cleanup failed: {str(e)}")
                await asyncio.sleep(3600)  # Retry in an hour if failed

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
            if os.path.exists(Config.CHROMA_PERSIST_DIR):
                logger.info("Loading existing vector store...")
                return Chroma(
                    persist_directory=Config.CHROMA_PERSIST_DIR,
                    embedding_function=self.embeddings
                )
            return Chroma(embedding_function=self.embeddings)
        except Exception as e:
            logger.error(f"Vector store initialization failed: {str(e)}")
            raise

    async def update_knowledge_base(self):
        if self.last_update and datetime.now() - self.last_update < self.update_interval:
            logger.info("Knowledge base update not required yet.")
            return

        all_content = []
        retry_count = Config.MAX_RETRIES

        for section, config in KNOWLEDGE_BASE.items():
            for attempt in range(retry_count):
                try:
                    content = await self._fetch_content(config['url'], config['patterns'])
                    if not content and section in self.backup_content:
                        content = self.backup_content[section]
                    if content:
                        self.redis.hset(f"knowledge:{section}", mapping=content)
                        self.redis.expire(f"knowledge:{section}", Config.REDIS_KEY_TTL)
                        all_content.extend(content.values())
                        break
                except Exception as e:
                    logger.error(f"Error updating {section} (attempt {attempt + 1}): {str(e)}")
                    if attempt == retry_count - 1:
                        # On final attempt, try to use backup content
                        if section in self.backup_content:
                            all_content.extend(self.backup_content[section].values())

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

            # Create backup before updating
            if os.path.exists(Config.CHROMA_PERSIST_DIR):
                backup_dir = f"{Config.CHROMA_PERSIST_DIR}_backup"
                os.system(f"cp -r {Config.CHROMA_PERSIST_DIR} {backup_dir}")

            try:
                vectorstore = Chroma.from_texts(
                    texts,
                    self.embeddings,
                    persist_directory=Config.CHROMA_PERSIST_DIR
                )
                
                self.vectorstore = vectorstore

                # Remove backup after successful update
                if os.path.exists(f"{Config.CHROMA_PERSIST_DIR}_backup"):
                    os.system(f"rm -r {Config.CHROMA_PERSIST_DIR}_backup")
            except Exception as e:
                # Restore from backup if update failed
                if os.path.exists(f"{Config.CHROMA_PERSIST_DIR}_backup"):
                    os.system(f"rm -r {Config.CHROMA_PERSIST_DIR}")
                    os.system(f"mv {Config.CHROMA_PERSIST_DIR}_backup {Config.CHROMA_PERSIST_DIR}")
                raise e

        except Exception as e:
            logger.error(f"Vector store update failed: {str(e)}")
            raise

    def get_relevant_content(self, query: str) -> List[str]:
        try:
            docs = self.vectorstore.similarity_search(query, k=3)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Error retrieving content: {str(e)}")
            return []

    def get_vector_store_size(self) -> int:
        try:
            return len(self.vectorstore.get())
        except Exception as e:
            logger.error(f"Error getting vector store size: {str(e)}")
            return 0


class ConversationManager:
    def __init__(self, knowledge_manager: KnowledgeManager, redis_client: Redis):
        self.knowledge_manager = knowledge_manager
        self.memory = RedisConversationMemory(redis_client)
        self.llm = ChatOpenAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
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
Previous conversation:
{chat_history}

Question: {question}

Provide a response that:
1. Directly addresses the question
2. References specific Artisan features
3. Includes relevant examples
4. Suggests next steps when appropriate

Response:""",
            input_variables=["context", "chat_history", "question"]
        )

    def _validate_history(self, history: List[Dict]) -> bool:
        """Validate that history contains required fields and proper formatting."""
        try:
            for msg in history:
                if not all(key in msg for key in ["role", "content", "timestamp"]):
                    logger.error(f"Invalid message format in history: {msg}")
                    return False
                if msg["role"] not in ["user", "assistant"]:
                    logger.error(f"Invalid role in message: {msg['role']}")
                    return False
                # Validate timestamp format
                datetime.fromisoformat(msg["timestamp"])
            return True
        except Exception as e:
            logger.error(f"History validation failed: {str(e)}")
            return False

    def get_conversation_chain(self, conversation_id: str) -> ConversationalRetrievalChain:
        try:
            # Initialize memory to store conversation pairs
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=Config.MEMORY_WINDOW_SIZE,
                output_key="answer"
            )

            # Retrieve and validate history from Redis
            history = self.memory.get_history(conversation_id)
            if not self._validate_history(history):
                logger.warning(f"Invalid history format for conversation {conversation_id}, starting fresh")
                history = []

            # Pair messages with improved error handling
            user_msg = None
            for msg in sorted(history, key=lambda x: x["timestamp"]):
                try:
                    if msg["role"] == "user":
                        # If we get a new user message and already had one, save the old one
                        if user_msg:
                            memory.save_context({"input": user_msg}, {"answer": ""})
                        user_msg = msg["content"]
                    elif msg["role"] == "assistant" and user_msg:
                        memory.save_context({"input": user_msg}, {"answer": msg["content"]})
                        user_msg = None
                except Exception as e:
                    logger.error(f"Error processing message in conversation {conversation_id}: {str(e)}")
                    continue

            # Handle any remaining unpaired user message
            if user_msg:
                memory.save_context({"input": user_msg}, {"answer": ""})

            # Update the TTL for the conversation key
            try:
                key = f"chat:{conversation_id}"
                self.memory.redis.expire(key, Config.REDIS_KEY_TTL)
            except Exception as e:
                logger.error(f"Failed to update TTL for conversation {conversation_id}: {str(e)}")

            # Create and return the conversational retrieval chain
            return ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.knowledge_manager.vectorstore.as_retriever(),
                memory=memory,
                combine_docs_chain_kwargs={"prompt": self.prompt},
                return_source_documents=True,
                verbose=True
            )
        except Exception as e:
            logger.error(f"Failed to create conversation chain for {conversation_id}: {str(e)}")
            raise


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
    if "data" in response.lower():
        actions.append("Explore available data sets")
    if "integration" in response.lower():
        actions.append("Visit integration documentation")
    return actions[:3]

app = FastAPI(
    title="Artisan AI Assistant",
    description="AI-powered assistant for Artisan platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware with configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize Services
redis_client = Redis.from_url(Config.REDIS_URL, decode_responses=True)
knowledge_manager = KnowledgeManager(redis_client)
conversation_manager = ConversationManager(knowledge_manager, redis_client)
metrics_collector = MetricsCollector(redis_client)

# Startup timestamp for uptime calculation
startup_time = datetime.now()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = datetime.now()
    request.state.conversation_id = "unknown"
   
    if request.url.path == "/chat" and request.method == "POST":
        try:
            body = await request.json()
            request.state.conversation_id = body.get('conversation_id', 'unknown')
        except Exception:
            pass  # If parsing fails, keep it as 'unknown'

    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    
    await metrics_collector.record_request(request.state.conversation_id, process_time)

    response.headers["X-Process-Time"] = str(process_time)
    return response



@app.post("/chat", response_model=ChatResponse)
@limiter.limit(Config.RATE_LIMIT)
async def chat(
    request: Request,                      # Request object for middlewares
    chat_request: ChatRequest,             # Request body data
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key) # Default parameter
):
    try:
        # Save the user's message using the body parameter 'chat_request'
        await conversation_manager.memory.save_message(
            chat_request.conversation_id,
            "user",
            chat_request.message
        )

        # Get response using the conversation chain
        chain = conversation_manager.get_conversation_chain(chat_request.conversation_id)
        result = chain({"question": chat_request.message})

        # Save the assistant's response
        await conversation_manager.memory.save_message(
            chat_request.conversation_id,
            "assistant",
            result["answer"]
        )

        # Schedule knowledge base update in background
        background_tasks.add_task(knowledge_manager.update_knowledge_base)

        return ChatResponse(
            response=result["answer"],
            sources=[doc.page_content[:200] for doc in result.get("source_documents", [])],
            confidence=min(len(result.get("source_documents", [])) / 3.0, 1.0),
            suggested_actions=_generate_suggested_actions(result["answer"]),
            related_topics=_extract_related_topics(chat_request.message, result["answer"])
        )
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/health", response_model=HealthCheckResponse)
async def health_check(api_key: str = Depends(verify_api_key)):
    process = psutil.Process()
    memory_info = process.memory_info()

    return HealthCheckResponse(
        status="healthy",
        vector_store=bool(knowledge_manager.vectorstore),
        vector_store_size=knowledge_manager.get_vector_store_size(),
        last_update=knowledge_manager.last_update,
        redis_connected=redis_client.ping(),
        memory_usage={
            "rss": memory_info.rss,
            "vms": memory_info.vms,
            "shared": memory_info.shared,
        },
        uptime=(datetime.now() - startup_time).total_seconds()
    )


@app.get("/metrics")
async def get_metrics(api_key: str = Depends(verify_api_key)):
    return metrics_collector.get_metrics()


@app.get("/")
async def root():
    return {
        "message": "Welcome to the Artisan AI Assistant API!",
        "version": "2.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "metrics": "/metrics"
    }


@app.get("/history/count/{conversation_id}")
async def get_history_count(
        conversation_id: str,
        api_key: str = Depends(verify_api_key)
):
    history = conversation_manager.memory.get_history(conversation_id)
    return {"message_count": len(history)}


@app.on_event("startup")
async def startup_event():
    # Initialize knowledge base
    await knowledge_manager.update_knowledge_base()
    logger.info("Application started, knowledge base initialized")



@app.on_event("shutdown")
async def shutdown_event():
    # Clean up resources
    redis_client.close()
    logger.info("Application shutdown, resources cleaned up")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))  # Add this line
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,  # Use port variable
        log_level="info",
        reload=True if os.getenv("ENVIRONMENT") == "development" else False
    )
