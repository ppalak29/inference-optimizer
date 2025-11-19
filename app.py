from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
from collections import Counter
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domains
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="templates")

# Analysis functions (from your old code)
def should_downgrade_to_gpt35_analysis(log):
    """Analysis version - determine if call could use GPT-3.5"""
    prompt = log['prompt'].lower()
    tokens = log.get('tokens', 0)
    
    if tokens < 100:
        return True
    
    if any(prompt.startswith(q) for q in ['what is', 'how do', 'why', 'when', 'who', 'where']):
        return True
    
    if 'translate' in prompt or 'translation' in prompt:
        return True
    
    if re.match(r'^[\d\+\-\*/\s]+$', prompt):
        return True
    
    return False

def analyze_data(logs):
    if not logs:
        return {
            'total_cost': 0,
            'total_calls': 0,
            'cache_savings': 0,
            'routing_savings': 0,
            'prompt_opt_savings': 0,
            'batch_savings': 0,
            'total_savings': 0,
            'savings_percent': 0,
            'duplicate_prompts': [],
            'downgradeable_calls': [],
            'verbose_prompts': [],
            'batchable_calls': []
        }
    
    total_cost = sum(float(log['cost']) for log in logs)
    total_calls = len(logs)
    
    # 1. CACHING
    prompts = [log['prompt'][:50] for log in logs]
    prompt_counts = Counter(prompts)
    
    duplicate_prompts = []
    cache_savings = 0
    for prompt, count in prompt_counts.items():
        if count > 1:
            matching_logs = [log for log in logs if log['prompt'][:50] == prompt]
            original_cost = float(matching_logs[0]['cost'])
            savings = original_cost * (count - 1)
            cache_savings += savings
            
            duplicate_prompts.append({
                'prompt': prompt,
                'count': count,
                'savings': round(savings, 2),
                'cost_per_call': round(original_cost, 4)
            })
    
    duplicate_prompts.sort(key=lambda x: x['savings'], reverse=True)
    
    # 2. MODEL ROUTING
    downgradeable_calls = []
    routing_savings = 0
    
    for log in logs:
        if 'gpt-4' in log['model'].lower() and should_downgrade_to_gpt35_analysis(log):
            savings = float(log['cost']) * 0.9
            routing_savings += savings
            
            downgradeable_calls.append({
                'prompt': log['prompt'][:60],
                'current_cost': round(float(log['cost']), 4),
                'new_cost': round(float(log['cost']) * 0.1, 4),
                'savings': round(savings, 4),
                'reason': 'Simple query' if log.get('tokens', 0) < 100 else 'Short prompt'
            })
    
    downgradeable_calls.sort(key=lambda x: x['savings'], reverse=True)
    
    # 3. PROMPT OPTIMIZATION
    verbose_prompts = []
    for log in logs:
        prompt_len = len(log['prompt'])
        if prompt_len > 500:
            verbose_prompts.append({
                'prompt_preview': log['prompt'][:80] + '...',
                'length': prompt_len,
                'cost': log['cost'],
                'timestamp': log['timestamp']
            })
    verbose_prompts.sort(key=lambda x: x['cost'], reverse=True)
    prompt_opt_savings = sum(v['cost'] * 0.3 for v in verbose_prompts)
    
    # 4. BATCHING (simplified for now)
    batch_savings = total_cost * 0.12
    
    total_savings = cache_savings + routing_savings + prompt_opt_savings + batch_savings
    
    return {
        'total_cost': round(total_cost, 2),
        'total_calls': total_calls,
        'cache_savings': round(cache_savings, 2),
        'routing_savings': round(routing_savings, 2),
        'prompt_opt_savings': round(prompt_opt_savings, 2),
        'batch_savings': round(batch_savings, 2),
        'total_savings': round(total_savings, 2),
        'savings_percent': round((total_savings/total_cost)*100, 1) if total_cost > 0 else 0,
        'duplicate_prompts': duplicate_prompts[:10],
        'downgradeable_calls': downgradeable_calls[:10],
        'verbose_prompts': verbose_prompts[:10],
        'batchable_calls': []
    }

# ROUTES

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request):
    return templates.TemplateResponse("analyze.html", {"request": request})

@app.post("/analyze")
async def analyze_upload(file: UploadFile = File(...)):
    """Upload CSV and analyze"""
    logs = []
    
    if file.filename.endswith('.csv'):
        content = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(content))
        
        for _, row in df.iterrows():
            logs.append({
                'timestamp': str(row.get('timestamp', '')),
                'prompt': str(row.get('prompt', ''))[:200],
                'model': str(row.get('model', 'gpt-3.5-turbo')),
                'tokens': int(row.get('n_context_tokens_total', 0)) + int(row.get('n_generated_tokens_total', 0)),
                'cost': float(row.get('cost', 0))
            })
    elif file.filename.endswith('.json'):
        content = await file.read()
        for line in content.decode('utf-8').strip().split('\n'):
            if line.strip():
                logs.append(json.loads(line))
    
    data = analyze_data(logs)
    return JSONResponse(data)

@app.get("/live", response_class=HTMLResponse)
async def live_dashboard(request: Request):
    return templates.TemplateResponse("live.html", {"request": request})

@app.get("/docs", response_class=HTMLResponse)
async def docs(request: Request):
    return templates.TemplateResponse("docs.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)