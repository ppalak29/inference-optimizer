#!/usr/bin/env python3
"""
Local Analysis Tool - Inference Optimizer

Analyzes your OpenAI usage CSV locally without uploading to any server.
Your data never leaves your machine.

Usage:
    python analyze_local.py your_usage.csv
"""

import sys
import json
import pandas as pd
from collections import Counter
import re

def should_downgrade(prompt, tokens):
    """Check if prompt could use GPT-3.5 instead of GPT-4"""
    prompt_lower = prompt.lower()
    
    if tokens < 100:
        return True
    
    if any(prompt_lower.startswith(q) for q in ['what is', 'what are', 'how do', 'why', 'when', 'who', 'where']):
        return True
    
    if 'translate' in prompt_lower or 'translation' in prompt_lower:
        return True
    
    if re.match(r'^[\d\+\-\*/\s]+$', prompt_lower):
        return True
    
    return False

def analyze_csv(filename):
    """Analyze OpenAI usage CSV"""
    print("\n" + "="*70)
    print(" "*20 + "INFERENCE OPTIMIZER")
    print(" "*22 + "Local Analysis")
    print("="*70)
    print(f"\n📁 Analyzing: {filename}")
    print("🔒 Running locally - your data stays on your machine\n")
    
    # Read CSV
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    # Parse logs
    logs = []
    for _, row in df.iterrows():
        logs.append({
            'timestamp': str(row.get('timestamp', '')),
            'prompt': str(row.get('prompt', ''))[:200],
            'model': str(row.get('model', 'gpt-3.5-turbo')),
            'tokens': int(row.get('n_context_tokens_total', 0)) + int(row.get('n_generated_tokens_total', 0)),
            'cost': float(row.get('cost', 0))
        })
    
    if not logs:
        print("❌ No data found in CSV")
        return
    
    total_cost = sum(log['cost'] for log in logs)
    total_calls = len(logs)
    
    print(f"📊 Found {total_calls:,} API calls")
    print(f"💰 Total spend: ${total_cost:,.2f}\n")
    
    # 1. CACHING ANALYSIS
    print("-"*70)
    print("1. 💾 CACHING OPPORTUNITIES")
    print("-"*70)
    
    prompts = [log['prompt'][:50] for log in logs]
    prompt_counts = Counter(prompts)
    duplicates = {p: c for p, c in prompt_counts.items() if c > 1}
    
    cache_savings = 0
    for prompt, count in duplicates.items():
        original_cost = next(log['cost'] for log in logs if log['prompt'][:50] == prompt)
        cache_savings += original_cost * (count - 1)
    
    print(f"Found {len(duplicates)} unique prompts called multiple times")
    print(f"Total duplicate calls: {sum(count - 1 for count in duplicates.values())}")
    print(f"💰 Potential savings: ${cache_savings:,.2f}")
    
    if duplicates:
        print("\nTop 5 duplicates:")
        sorted_dups = sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:5]
        for prompt, count in sorted_dups:
            print(f"  • '{prompt}' - called {count}x")
    
    # 2. MODEL ROUTING ANALYSIS
    print("\n" + "-"*70)
    print("2. 🔄 MODEL ROUTING OPPORTUNITIES")
    print("-"*70)
    
    downgradeable = []
    routing_savings = 0
    
    for log in logs:
        if 'gpt-4' in log['model'].lower() and should_downgrade(log['prompt'], log['tokens']):
            savings = log['cost'] * 0.9
            routing_savings += savings
            downgradeable.append({
                'prompt': log['prompt'][:60],
                'cost': log['cost'],
                'savings': savings
            })
    
    print(f"Found {len(downgradeable)} GPT-4 calls that could use GPT-3.5")
    print(f"💰 Potential savings: ${routing_savings:,.2f}")
    
    if downgradeable:
        print("\nTop 5 downgradeable calls:")
        sorted_down = sorted(downgradeable, key=lambda x: x['savings'], reverse=True)[:5]
        for call in sorted_down:
            print(f"  • '{call['prompt']}' - save ${call['savings']:.4f}")
    
    # 3. PROMPT OPTIMIZATION
    print("\n" + "-"*70)
    print("3. ✂️  PROMPT OPTIMIZATION")
    print("-"*70)
    
    verbose = [log for log in logs if len(log['prompt']) > 500]
    prompt_savings = sum(log['cost'] * 0.3 for log in verbose)
    
    print(f"Found {len(verbose)} verbose prompts (>500 chars)")
    print(f"💰 Potential savings: ~${prompt_savings:,.2f}")
    
    # 4. BATCHING
    print("\n" + "-"*70)
    print("4. 📦 REQUEST BATCHING")
    print("-"*70)
    
    batch_savings = total_cost * 0.12
    print(f"💰 Potential savings: ~${batch_savings:,.2f}")
    print("(Combine sequential similar requests)")
    
    # TOTAL
    total_savings = cache_savings + routing_savings + prompt_savings + batch_savings
    
    print("\n" + "="*70)
    print("💰 TOTAL POTENTIAL SAVINGS")
    print("="*70)
    print(f"\nCurrent spend:     ${total_cost:,.2f}/month")
    print(f"Potential savings: ${total_savings:,.2f}/month ({total_savings/total_cost*100:.1f}%)")
    print(f"New spend:         ${total_cost - total_savings:,.2f}/month")
    
    print("\n" + "="*70)
    print("🚀 READY TO SAVE THIS MONEY?")
    print("="*70)
    print("\nInference Optimizer implements these optimizations automatically:")
    print("  • Automatic caching (Redis)")
    print("  • Smart model routing (GPT-4 → GPT-3.5 when appropriate)")
    print("  • Real-time savings dashboard")
    print("  • One-line code integration")
    
    print("\n📧 Get started: https://your-site.com")
    print("="*70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n❌ Error: No file specified")
        print("\nUsage:")
        print("  python analyze_local.py your_usage.csv")
        print("\nDownload your usage CSV from:")
        print("  https://platform.openai.com/usage")
        print("\n🔒 Note: Analysis runs locally. Your data never leaves your machine.\n")
        sys.exit(1)
    
    analyze_csv(sys.argv[1])