# Artisan AI Assistant

A FastAPI backend for the Artisan AI Assistant.

## Features

- Chat API endpoint at \`/chat\`
- Health check endpoint at \`/health\`
- Conversation history management using Redis
- Knowledge base integration using a Chroma vector store
- Ready for deployment on Heroku, Fly.io, or Render

## Setup

1. **Clone the Repository:**
   \`\`\`bash
   git clone https://github.com/your_username/artisan-ai-assistant.git
   cd artisan-ai-assistant
   \`\`\`

2. **Create a Virtual Environment and Install Dependencies:**
   \`\`\`bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   \`\`\`

3. **Set Up Environment Variables:**

   Create a \`.env\` file in the root directory and add:
   \`\`\`
   OPENAI_API_KEY=your_openai_api_key_here
   REDIS_URL=redis://localhost:6379
   \`\`\`

4. **Run the Application Locally:**
   \`\`\`bash
   uvicorn main:app --reload
   \`\`\`

## Deployment

For example, to deploy on Heroku:

1. **Login and Create a Heroku App:**
   \`\`\`bash
   heroku login
   heroku create artisan-ai-assistant
   \`\`\`

2. **Set Environment Variables on Heroku:**
   \`\`\`bash
   heroku config:set OPENAI_API_KEY=your_openai_api_key_here
   heroku config:set REDIS_URL=your_redis_url_here
   \`\`\`

3. **Deploy:**
   \`\`\`bash
   git push heroku main
   \`\`\`

## License

MIT License
