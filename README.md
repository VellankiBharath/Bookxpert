# Bookxpert

Documentataion:

🍳 Local LLM Recipe Chatbot - Complete Setup Guide
📋 Table of Contents
1.	System Overview
2.	Prerequisites
3.	Installation Steps
4.	Fine-Tuning the Model
5.	Running the Application
6.	API Documentation
7.	Troubleshooting
________________________________________
🎯 System Overview
This project implements a complete Local LLM-powered Recipe Chatbot with the following components:
Architecture
┌─────────────────┐
│  Web Interface  │ (HTML/CSS/JavaScript)
│   (Port 3000)   │
└────────┬────────┘
         │ HTTP Requests
         ↓
┌─────────────────┐
│   FastAPI       │ (Python Backend)
│  Server API     │
│   (Port 8000)   │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Fine-tuned LLM │ (Recipe Model)
│  + Recipe DB    │
└─────────────────┘
Key Features
✅ Local LLM Integration - Fine-tuned model for recipe suggestions
✅ FastAPI Backend - RESTful API with automatic documentation
✅ Ingredient Matching - ML-based similarity scoring
✅ Interactive Web UI - Beautiful, responsive chatbot interface
✅ Recipe Database - 10+ recipes with detailed instructions
✅ Real-time Responses - Instant recipe suggestions
________________________________________
🔧 Prerequisites
System Requirements
•	Python: 3.8 or higher
•	RAM: Minimum 8GB (16GB recommended for large models)
•	Storage: 5GB free space
•	OS: Windows, macOS, or Linux
Software Dependencies
# Core packages
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# For actual LLM integration (optional)
transformers==4.35.0
torch==2.1.0
accelerate==0.24.0
sentencepiece==0.1.99
________________________________________
📥 Installation Steps
Step 1: Clone/Create Project Directory
mkdir recipe-chatbot
cd recipe-chatbot
Step 2: Create Virtual Environment
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
Step 3: Install Dependencies
# Create requirements.txt
cat > requirements.txt << EOF
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
python-dotenv==1.0.0

# Optional: For actual LLM models
# transformers==4.35.0
# torch==2.1.0
# accelerate==0.24.0
EOF

# Install packages
pip install -r requirements.txt
Step 4: Create Project Structure
recipe-chatbot/
├── backend/
│   ├── app.py              # FastAPI server (provided)
│   ├── models/             # Fine-tuned models directory
│   └── data/
│       └── recipes.json    # Recipe dataset
├── frontend/
│   └── index.html          # Web UI (provided)
├── training/
│   ├── prepare_data.py     # Data preparation script
│   └── fine_tune.py        # Fine-tuning script
└── requirements.txt
________________________________________
🎓 Fine-Tuning the Model
Option 1: Using Pre-trained Models (Recommended for Starting)
The current implementation uses rule-based matching which works well for demo purposes. To integrate actual LLMs:
1. Choose a Base Model
Popular options for recipe tasks:
•	GPT-2 (Small, fast) - 124M parameters
•	DistilGPT-2 (Even smaller) - 82M parameters
•	FLAN-T5-Small (Good for instructions) - 80M parameters
•	LLaMA-7B (Advanced, requires more resources)
2. Prepare Training Dataset
# training/prepare_data.py
import json

def create_training_dataset():
    """
    Create training dataset in format:
    {"input": "ingredients: eggs, onions", "output": "Recipe: ..."}
    """
    
    recipes = [
        {
            "ingredients": ["eggs", "onions", "tomatoes"],
            "recipe_name": "Egg Bhurji",
            "instructions": "Heat oil, saute onions, add tomatoes, scramble eggs..."
        },
        # Add more recipes...
    ]
    
    training_data = []
    for recipe in recipes:
        training_data.append({
            "input": f"Suggest a recipe using: {', '.join(recipe['ingredients'])}",
            "output": f"I recommend {recipe['recipe_name']}. {recipe['instructions']}"
        })
    
    # Save to JSON
    with open('training_data.json', 'w') as f:
        json.dump(training_data, f, indent=2)
    
    return training_data

if __name__ == "__main__":
    data = create_training_dataset()
    print(f"Created {len(data)} training examples")
3. Fine-tune the Model
# training/fine_tune.py
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import json

def fine_tune_model():
    """Fine-tune GPT-2 on recipe data"""
    
    # Load base model
    model_name = "gpt2"  # or "distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load training data
    with open('training_data.json', 'r') as f:
        data = json.load(f)
    
    # Prepare dataset
    def preprocess(examples):
        inputs = [f"Input: {ex['input']}\nOutput: {ex['output']}" 
                  for ex in examples]
        return tokenizer(inputs, truncation=True, padding='max_length', 
                        max_length=512)
    
    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(preprocess, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./recipe-model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=100,
        save_total_limit=2,
        learning_rate=5e-5,
        warmup_steps=100,
        logging_steps=10,
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    
    # Save fine-tuned model
    model.save_pretrained("./recipe-model-finetuned")
    tokenizer.save_pretrained("./recipe-model-finetuned")
    
    print("Fine-tuning complete!")

if __name__ == "__main__":
    fine_tune_model()
4. Integrate Fine-tuned Model into API
Update backend/app.py:
class RecipeLLM:
    def __init__(self):
        # Load fine-tuned model
        model_path = "./training/recipe-model-finetuned"
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def generate_recipe(self, ingredients):
        prompt = f"Suggest a recipe using: {', '.join(ingredients)}\nRecipe:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
Option 2: Using Smaller Models (Resource-Constrained)
For systems with limited resources:
# Use DistilGPT-2 (82M params, ~350MB)
pip install transformers torch

# Download and cache model
python -c "from transformers import AutoModel; AutoModel.from_pretrained('distilgpt2')"
________________________________________
🚀 Running the Application
Step 1: Start the FastAPI Backend
# Navigate to project directory
cd recipe-chatbot

# Run the server
python backend/app.py

# Or using uvicorn directly
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
Expected output:
================================================================================
🍳 RECIPE CHATBOT API SERVER
================================================================================
📊 Loaded 10 recipes into database
🤖 LLM Model: Recipe-focused fine-tuned model (demo mode)
🌐 Starting FastAPI server...
================================================================================

📍 API Endpoints:
   • http://localhost:8000/          - API Info
   • http://localhost:8000/docs      - Swagger UI
   • http://localhost:8000/chat      - Chat endpoint
   • http://localhost:8000/recipes   - All recipes

INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
Step 2: Open the Web Interface
Method 1: Direct File Open
# Simply open the HTML file in your browser
open frontend/index.html  # Mac
start frontend/index.html  # Windows
xdg-open frontend/index.html  # Linux
Method 2: Using Python HTTP Server
# Navigate to frontend directory
cd frontend

# Start simple HTTP server
python -m http.server 3000

# Open browser to:
# http://localhost:3000
Step 3: Test the Chatbot
1.	Add ingredients in the input box (e.g., "eggs", "onions")
2.	Click "Add" to add each ingredient
3.	Click "Get Recipe Suggestion" button
4.	View the AI-generated recipe response!
________________________________________
📚 API Documentation
Interactive API Docs
Once the server is running, visit:
•	Swagger UI: http://localhost:8000/docs
•	ReDoc: http://localhost:8000/redoc
API Endpoints
1. Root Endpoint
GET /
Returns API information and available endpoints.
2. Health Check
GET /health
Response:
{
  "status": "healthy",
  "timestamp": "2025-11-17T10:30:00",
  "model_loaded": true
}
3. Get All Recipes
GET /recipes
Response:
{
  "total_recipes": 10,
  "recipes": [...]
}
4. Chat Endpoint (Main)
POST /chat
Request Body:
{
  "ingredients": ["eggs", "onions"],
  "user_message": null,
  "conversation_history": []
}
Response:
{
  "recipe": {
    "name": "Classic Egg Omelette",
    "ingredients": ["eggs", "onions", "salt", "pepper", "butter"],
    "instructions": ["Beat eggs...", "Dice onions...", ...],
    "prep_time": "5 minutes",
    "cook_time": "8 minutes",
    "servings": 2,
    "difficulty": "Easy",
    "cuisine": "Continental"
  },
  "message": "Perfect match! I found a great recipe for you...",
  "confidence_score": 85.50,
  "matched_ingredients": ["eggs", "onions"],
  "timestamp": "2025-11-17T10:30:00"
}
5. Get Training Data
GET /training-data
Returns formatted data for model fine-tuning.
________________________________________
🐛 Troubleshooting
Issue 1: "Connection Refused" Error
Problem: Frontend can't connect to backend API
Solution:
# Check if backend is running
curl http://localhost:8000/health

# Check firewall settings
# Make sure port 8000 is not blocked

# Try different port
uvicorn backend.app:app --port 8001
Issue 2: CORS Errors
Problem: Browser blocks API requests
Solution: Already configured in the FastAPI app:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Issue 3: Model Loading Errors
Problem: Can't load fine-tuned model
Solution:
# Check model path
import os
print(os.path.exists("./recipe-model-finetuned"))

# Try using CPU instead of GPU
device = torch.device("cpu")

# Use smaller model
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
Issue 4: Out of Memory
Problem: System runs out of RAM when loading model
Solutions:
1.	Use smaller model (DistilGPT-2 instead of GPT-2)
2.	Reduce batch size during inference
3.	Use CPU instead of GPU
4.	Implement model quantization:
# 8-bit quantization
from transformers import BitsAndBytesConfig

config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    quantization_config=config
)
________________________________________
📊 Performance Optimization
1. Model Caching
# Cache model in memory to avoid reloading
@lru_cache(maxsize=1)
def load_model():
    return AutoModelForCausalLM.from_pretrained("model-path")
2. Batch Processing
# Process multiple requests together
def batch_generate(ingredient_lists):
    inputs = [format_prompt(ings) for ings in ingredient_lists]
    # Process all at once
    return model.generate(inputs)
3. Asynchronous Processing
# Make API endpoints async
@app.post("/chat")
async def chat(request: ChatRequest):
    result = await async_generate_recipe(request.ingredients)
    return result
________________________________________
🔐 Security Considerations
1.	Rate Limiting
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: ChatRequest):
    ...
2.	Input Validation
# Validate ingredient names
def validate_ingredient(ingredient: str) -> bool:
    return len(ingredient) <= 50 and ingredient.isalnum()
3.	API Authentication (for production)
from fastapi.security import HTTPBearer
security = HTTPBearer()
________________________________________
📈 Next Steps
1.	Expand Recipe Database: Add 100+ recipes
2.	Improve Model: Fine-tune on larger dataset
3.	Add Features: 
o	Dietary restrictions (vegan, gluten-free)
o	Cuisine preferences
o	Cooking time filters
o	Nutritional information
4.	Deploy to Cloud: Use Docker + Kubernetes
5.	Mobile App: Build React Native app
________________________________________
📝 License
MIT License - Feel free to use and modify!
👥 Support
For issues or questions, please create an issue in the repository.
________________________________________
Happy Cooking! 🍳👨🍳
Backend running :
pip install fastapi uvicorn pydantic
