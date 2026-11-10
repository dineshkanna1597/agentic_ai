from langchain_litellm import ChatLiteLLMRouter,LiteLLMEmbeddingsRouter
from dotenv import load_dotenv
from litellm import Router
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError('GROQ_API_KEY is not configured')

llm_router = Router(
    model_list=[
        {
            'model_name':'reasoning',
            'litellm_params':{
                'model':'groq/openai/gpt-oss-120b',
                'api_key':groq_api_key,
                'timeout':30,
            },
        },
        {
            'model_name': 'reasoning_backup1',
            'litellm_params': {
                'model': 'groq/openai/gpt-oss-20b',
                'api_key': groq_api_key,
                'timeout': 30,
            },
        },
        {
            'model_name':'reasoning_backup2',
            'litellm_params':{
                'model':'ollama/gpt-oss-20b',
                'api_base':'http://localhost:11434',
                'timeout':1800,
            },
        },
    ],
    num_retries=2,
    retry_after=1,
    fallbacks=[
        {'reasoning':['reasoning_backup1','reasoning_backup2']}
    ],
)

embedding_router = Router(
    model_list=[
        {
            'model_name':'text-embedding',
            'litellm_params':{
                'model':'ollama/qwen3-embedding:8b',
                'api_base':'http://localhost:11434',
                'timeout':1800,
            },
        },
    ],
    num_retries=2,
    retry_after=1,
)

def get_chat_model():
    return ChatLiteLLMRouter(
        router=llm_router,
        model_name='reasoning',
        temperature=0,
        max_tokens=1024,
    )

def get_embedding_model():
    return LiteLLMEmbeddingsRouter(
        router=embedding_router,
        model='text-embedding',
        )