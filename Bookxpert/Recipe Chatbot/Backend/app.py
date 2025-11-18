"""
Recipe Chatbot Backend - FastAPI Server with Local LLM Integration
Supports both custom fine-tuned models and recipe matching system
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import json
from datetime import datetime

# For actual LLM integration (uncomment when using real models)
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

app = FastAPI(
    title="Recipe Chatbot API",
    description="Local LLM-powered recipe suggestion chatbot",
    version="1.0.0"
)

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== DATA MODELS ====================

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    ingredients: List[str]
    user_message: Optional[str] = None
    conversation_history: Optional[List[Dict]] = []


class Recipe(BaseModel):
    """Recipe data model"""
    name: str
    ingredients: List[str]
    instructions: List[str]
    prep_time: str
    cook_time: str
    servings: int
    difficulty: str
    cuisine: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    recipe: Optional[Recipe]
    message: str
    confidence_score: float
    matched_ingredients: List[str]
    timestamp: str


# ==================== RECIPE DATABASE ====================

RECIPE_DATABASE = [
    {
        "name": "Classic Egg Omelette",
        "ingredients": ["eggs", "onions", "salt", "pepper", "butter", "cheese"],
        "instructions": [
            "Beat 2-3 eggs in a bowl with salt and pepper",
            "Dice onions finely",
            "Heat butter in a non-stick pan over medium heat",
            "Sauté onions until translucent",
            "Pour beaten eggs over onions",
            "Let cook for 2-3 minutes until edges set",
            "Add cheese if desired",
            "Fold omelette in half and serve hot"
        ],
        "prep_time": "5 minutes",
        "cook_time": "8 minutes",
        "servings": 2,
        "difficulty": "Easy",
        "cuisine": "Continental"
    },
    {
        "name": "Egg Bhurji (Indian Scrambled Eggs)",
        "ingredients": ["eggs", "onions", "tomatoes", "green chilies", "turmeric", "oil", "cilantro"],
        "instructions": [
            "Heat oil in a pan",
            "Add chopped onions and green chilies, sauté until golden",
            "Add diced tomatoes and cook until soft",
            "Add turmeric and salt",
            "Beat eggs and pour into the pan",
            "Scramble continuously until fully cooked",
            "Garnish with fresh cilantro",
            "Serve with bread or roti"
        ],
        "prep_time": "10 minutes",
        "cook_time": "10 minutes",
        "servings": 2,
        "difficulty": "Easy",
        "cuisine": "Indian"
    },
    {
        "name": "Egg Fried Rice",
        "ingredients": ["eggs", "rice", "onions", "soy sauce", "garlic", "vegetables", "oil"],
        "instructions": [
            "Cook rice and let it cool (day-old rice works best)",
            "Beat eggs and scramble in a wok, set aside",
            "Heat oil and sauté minced garlic and onions",
            "Add mixed vegetables and stir-fry",
            "Add cold rice and break up any clumps",
            "Add soy sauce and mix well",
            "Add scrambled eggs back and toss everything together",
            "Serve hot with additional soy sauce if needed"
        ],
        "prep_time": "15 minutes",
        "cook_time": "15 minutes",
        "servings": 4,
        "difficulty": "Medium",
        "cuisine": "Asian"
    },
    {
        "name": "Onion Rings",
        "ingredients": ["onions", "flour", "eggs", "breadcrumbs", "salt", "pepper", "oil"],
        "instructions": [
            "Slice onions into thick rings and separate",
            "Set up breading station: flour, beaten eggs, breadcrumbs",
            "Season flour with salt and pepper",
            "Coat onion rings in flour, dip in egg, then coat with breadcrumbs",
            "Heat oil to 350°F (175°C)",
            "Fry rings until golden brown, about 2-3 minutes per side",
            "Drain on paper towels",
            "Serve hot with dipping sauce"
        ],
        "prep_time": "20 minutes",
        "cook_time": "15 minutes",
        "servings": 4,
        "difficulty": "Medium",
        "cuisine": "American"
    },
    {
        "name": "Spanish Tortilla (Potato Omelette)",
        "ingredients": ["eggs", "potatoes", "onions", "olive oil", "salt"],
        "instructions": [
            "Peel and thinly slice potatoes and onions",
            "Heat generous olive oil in a pan",
            "Fry potatoes and onions slowly until tender (15-20 mins)",
            "Beat 6 eggs in a large bowl with salt",
            "Drain potatoes and onions, add to beaten eggs",
            "Heat 2 tbsp oil in a non-stick pan",
            "Pour mixture and cook on medium-low heat",
            "Flip using a plate when bottom is set",
            "Cook other side until golden",
            "Serve warm or at room temperature"
        ],
        "prep_time": "15 minutes",
        "cook_time": "30 minutes",
        "servings": 6,
        "difficulty": "Medium",
        "cuisine": "Spanish"
    },
    {
        "name": "Egg Curry",
        "ingredients": ["eggs", "onions", "tomatoes", "ginger", "garlic", "spices", "oil", "coconut milk"],
        "instructions": [
            "Boil eggs, peel and set aside",
            "Make paste of onions, ginger, and garlic",
            "Heat oil and fry the paste until golden",
            "Add tomato puree and cook until oil separates",
            "Add curry spices (turmeric, coriander, cumin, garam masala)",
            "Add coconut milk and water, simmer for 10 minutes",
            "Cut eggs in half and add to curry",
            "Simmer for 5 more minutes",
            "Garnish with cilantro and serve with rice or naan"
        ],
        "prep_time": "15 minutes",
        "cook_time": "25 minutes",
        "servings": 4,
        "difficulty": "Medium",
        "cuisine": "Indian"
    },
    {
        "name": "Caramelized Onion Pasta",
        "ingredients": ["onions", "pasta", "butter", "garlic", "parmesan", "white wine", "thyme"],
        "instructions": [
            "Slice onions thinly",
            "Melt butter in a large pan over low heat",
            "Add onions and cook slowly for 30-40 minutes, stirring occasionally",
            "Meanwhile, cook pasta according to package directions",
            "Add minced garlic to onions in last 5 minutes",
            "Deglaze pan with white wine",
            "Add cooked pasta to caramelized onions",
            "Toss with butter, fresh thyme, and parmesan",
            "Season with salt and pepper",
            "Serve immediately with extra parmesan"
        ],
        "prep_time": "10 minutes",
        "cook_time": "45 minutes",
        "servings": 4,
        "difficulty": "Medium",
        "cuisine": "Italian"
    },
    {
        "name": "Deviled Eggs",
        "ingredients": ["eggs", "mayonnaise", "mustard", "paprika", "salt", "pepper"],
        "instructions": [
            "Hard boil eggs for 10 minutes",
            "Cool in ice water and peel carefully",
            "Cut eggs in half lengthwise",
            "Remove yolks and place in a bowl",
            "Mash yolks with mayonnaise, mustard, salt, and pepper",
            "Spoon or pipe mixture back into egg white halves",
            "Sprinkle with paprika",
            "Chill before serving"
        ],
        "prep_time": "15 minutes",
        "cook_time": "10 minutes",
        "servings": 12,
        "difficulty": "Easy",
        "cuisine": "American"
    },
    {
        "name": "French Onion Soup",
        "ingredients": ["onions", "butter", "beef broth", "white wine", "bread", "cheese", "thyme"],
        "instructions": [
            "Slice 4-5 large onions thinly",
            "Melt butter in a large pot over medium-low heat",
            "Add onions and cook for 45 minutes until deeply caramelized",
            "Add minced garlic and cook 1 minute",
            "Deglaze with white wine",
            "Add beef broth and thyme, simmer 30 minutes",
            "Toast bread slices",
            "Ladle soup into oven-safe bowls",
            "Top with bread and grated Gruyere cheese",
            "Broil until cheese is melted and bubbly"
        ],
        "prep_time": "15 minutes",
        "cook_time": "90 minutes",
        "servings": 6,
        "difficulty": "Medium",
        "cuisine": "French"
    },
    {
        "name": "Egg Drop Soup",
        "ingredients": ["eggs", "chicken broth", "cornstarch", "soy sauce", "ginger", "scallions"],
        "instructions": [
            "Heat chicken broth in a pot with grated ginger",
            "Mix cornstarch with water to make a slurry",
            "Add cornstarch slurry to simmering broth to thicken slightly",
            "Beat eggs in a bowl",
            "While stirring broth in one direction, slowly drizzle in beaten eggs",
            "The eggs will form thin ribbons",
            "Add soy sauce to taste",
            "Garnish with chopped scallions",
            "Serve hot"
        ],
        "prep_time": "5 minutes",
        "cook_time": "10 minutes",
        "servings": 4,
        "difficulty": "Easy",
        "cuisine": "Chinese"
    }
]


# ==================== LLM MODEL CLASS ====================

class RecipeLLM:
    """
    Recipe-focused Language Model wrapper
    In production, this would load a fine-tuned model
    """
    
    def __init__(self):
        """Initialize the model"""
        # In production, load your fine-tuned model here:
        # self.model = AutoModelForCausalLM.from_pretrained("path/to/fine-tuned-model")
        # self.tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)
        
        print("Recipe LLM initialized (using rule-based matching for demo)")
    
    def preprocess_ingredients(self, ingredients: List[str]) -> List[str]:
        """Clean and normalize ingredient names"""
        return [ing.lower().strip() for ing in ingredients]
    
    def calculate_match_score(self, recipe_ingredients: List[str], 
                             user_ingredients: List[str]) -> tuple:
        """Calculate how well user ingredients match a recipe"""
        recipe_ings_lower = [ing.lower() for ing in recipe_ingredients]
        user_ings_lower = [ing.lower() for ing in user_ingredients]
        
        matched = []
        for user_ing in user_ings_lower:
            for recipe_ing in recipe_ings_lower:
                if user_ing in recipe_ing or recipe_ing in user_ing:
                    matched.append(user_ing)
                    break
        
        # Calculate score based on matches
        if len(user_ings_lower) == 0:
            return 0, []
        
        score = len(matched) / len(user_ings_lower)
        return score, matched
    
    def find_best_recipe(self, ingredients: List[str]) -> tuple:
        """Find the best matching recipe using ingredient matching"""
        ingredients = self.preprocess_ingredients(ingredients)
        
        if not ingredients:
            return None, 0, []
        
        best_recipe = None
        best_score = 0
        best_matched = []
        
        for recipe in RECIPE_DATABASE:
            score, matched = self.calculate_match_score(
                recipe['ingredients'], 
                ingredients
            )
            
            if score > best_score:
                best_score = score
                best_recipe = recipe
                best_matched = matched
        
        return best_recipe, best_score, best_matched
    
    def generate_response(self, ingredients: List[str]) -> Dict:
        """Generate a recipe suggestion based on ingredients"""
        recipe, score, matched = self.find_best_recipe(ingredients)
        
        if recipe is None:
            return {
                "recipe": None,
                "message": "I couldn't find any recipes with those ingredients. Try adding more common ingredients like eggs, onions, rice, or pasta!",
                "confidence_score": 0,
                "matched_ingredients": []
            }
        
        # Generate conversational message
        if score >= 0.8:
            confidence_msg = "Perfect match! "
        elif score >= 0.5:
            confidence_msg = "Good match! "
        else:
            confidence_msg = "Partial match - "
        
        message = (
            f"{confidence_msg}I found a great recipe for you: **{recipe['name']}**! "
            f"This {recipe['cuisine']} dish matches {len(matched)} of your ingredients "
            f"and takes about {recipe['prep_time']} to prep and {recipe['cook_time']} to cook. "
            f"It's rated as {recipe['difficulty']} difficulty. Would you like to try it?"
        )
        
        return {
            "recipe": recipe,
            "message": message,
            "confidence_score": round(score * 100, 2),
            "matched_ingredients": matched
        }


# Initialize the model
llm_model = RecipeLLM()


# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Recipe Chatbot API is running!",
        "version": "1.0.0",
        "endpoints": {
            "/chat": "POST - Get recipe suggestions",
            "/recipes": "GET - List all available recipes",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": True
    }


@app.get("/recipes")
async def get_all_recipes():
    """Get all available recipes in the database"""
    return {
        "total_recipes": len(RECIPE_DATABASE),
        "recipes": RECIPE_DATABASE
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - accepts ingredients and returns recipe suggestions
    """
    try:
        if not request.ingredients:
            raise HTTPException(
                status_code=400, 
                detail="Please provide at least one ingredient"
            )
        
        # Generate response using LLM
        response = llm_model.generate_response(request.ingredients)
        
        # Format response
        return ChatResponse(
            recipe=Recipe(**response["recipe"]) if response["recipe"] else None,
            message=response["message"],
            confidence_score=response["confidence_score"],
            matched_ingredients=response["matched_ingredients"],
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint for real-time responses
    (Useful for actual LLM token streaming)
    """
    response = llm_model.generate_response(request.ingredients)
    return response


# ==================== FINE-TUNING UTILITIES ====================

def prepare_training_data():
    """
    Prepare dataset for fine-tuning
    Format: Input (ingredients) -> Output (recipe)
    """
    training_data = []
    
    for recipe in RECIPE_DATABASE:
        # Create training examples
        training_example = {
            "input": f"Suggest a recipe using: {', '.join(recipe['ingredients'][:3])}",
            "output": f"I recommend {recipe['name']}. {' '.join(recipe['instructions'][:2])}"
        }
        training_data.append(training_example)
    
    return training_data


@app.get("/training-data")
async def get_training_data():
    """Endpoint to retrieve training data for fine-tuning"""
    return {
        "training_examples": prepare_training_data(),
        "total_examples": len(RECIPE_DATABASE)
    }


# ==================== SERVER STARTUP ====================

if __name__ == "__main__":
    print("="*80)
    print("🍳 RECIPE CHATBOT API SERVER")
    print("="*80)
    print(f"📊 Loaded {len(RECIPE_DATABASE)} recipes into database")
    print("🤖 LLM Model: Recipe-focused fine-tuned model (demo mode)")
    print("🌐 Starting FastAPI server...")
    print("="*80)
    print("\n📍 API Endpoints:")
    print("   • http://127.0.0.1:8000/          - API Info")
    print("   • http://127.0.0.1:8000/docs      - Swagger UI")
    print("   • http://127.0.0.1:8000/chat      - Chat endpoint")
    print("   • http://127.0.0.1:8000/recipes   - All recipes")
    print("\n" + "="*80 + "\n")
    
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000,
        log_level="info"
    )