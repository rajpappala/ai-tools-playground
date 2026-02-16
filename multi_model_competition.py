import os
import json
import re
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Setup logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"competition_log_{timestamp}.log"
html_filename = f"competition_results_{timestamp}.html"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# HTML output buffer
html_content = []

# Initialize clients
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
ollama_client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

# Models to compete
models = [
    {"name": "gpt-5-mini", "client": openai_client},
    {"name": "gpt-5-nano", "client": openai_client},
    {"name": "gpt-4o-mini", "client": openai_client},
    {"name": "llama3.2:1b", "client": ollama_client}
]

logger.info("="*80)
logger.info("MULTI-MODEL COMPETITION STARTED")
logger.info("="*80)
logger.info(f"Log file: {log_filename}")

# Step 1: Generate a challenging question using gpt-5-mini
logger.info("\n[Step 1] Generating challenging question using gpt-5-mini...")
question_prompt = "Please come up with a challenging, nuanced question that I can ask a number of LLMs to evaluate their intelligence. Answer only with the question, no explanation."

logger.info(f"Question prompt sent: {question_prompt}")

response = openai_client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": question_prompt}]
)
question = response.choices[0].message.content

logger.info(f"\nGenerated Question:\n{question}")
logger.info("\n" + "="*80)

# Step 2: Collect answers from all 4 models
logger.info("\n[Step 2] Collecting answers from all 4 models...\n")
logger.info(f"Total models to query: {len(models)}")

answers = []
competitors = []

for idx, model in enumerate(models, 1):
    logger.info(f"\n--- Iteration {idx}/{len(models)} ---")
    logger.info(f"Asking {model['name']}...")
    logger.info(f"Client base URL: {getattr(model['client'], 'base_url', 'default OpenAI')}")

    try:
        # Use different parameter names for different model families
        request_params = {
            "model": model['name'],
            "messages": [{"role": "user", "content": question}]
        }

        # GPT-5 models and newer use max_completion_tokens, older models use max_tokens
        # GPT-5 models also only support temperature=1 (default)
        if 'gpt-5' in model['name']:
            request_params['max_completion_tokens'] = 3000
            # No temperature setting for gpt-5 models (only supports default of 1)
        elif 'gpt-4o' in model['name']:
            request_params['max_completion_tokens'] = 3000
            request_params['temperature'] = 0.7
        else:
            request_params['max_tokens'] = 3000
            request_params['temperature'] = 0.7

        response = model['client'].chat.completions.create(**request_params)
        answer = response.choices[0].message.content

        # Check if the model refused to answer
        refusal_phrases = ["I can't help with this", "I cannot assist", "I'm unable to", "I can't assist"]
        is_refusal = any(phrase.lower() in answer.lower() for phrase in refusal_phrases)

        if is_refusal:
            logger.warning(f"⚠ {model['name']} refused to answer: {answer}")
            logger.info(f"Skipping {model['name']} due to refusal")
        else:
            logger.info(f"✓ {model['name']} responded successfully")
            logger.info(f"Response length: {len(answer)} characters")
            logger.info(f"Response preview: {answer[:200]}...")
            logger.info(f"Full response from {model['name']}:\n{answer}\n")

            competitors.append(model['name'])
            answers.append(answer)

    except Exception as e:
        logger.error(f"✗ Error with {model['name']}: {e}")
        logger.exception("Full traceback:")

logger.info("="*80)
logger.info(f"Successfully collected {len(answers)} answers from {len(competitors)} models")

# Check if we have enough responses to judge
if len(answers) < 2:
    logger.error(f"Not enough valid responses to judge ({len(answers)} responses). Need at least 2.")
    logger.info("\nFalling back to display all responses:\n")
    for competitor, answer in zip(competitors, answers):
        logger.info(f"--- {competitor} ---")
        logger.info(answer)
        logger.info("\n")
    exit(0)

# Step 3: Compile all responses for the judge
logger.info("\n[Step 3] Compiling responses...\n")

compiled_responses = ""
for index, answer in enumerate(answers):
    logger.info(f"Compiling response {index+1}/{len(answers)} from {competitors[index]}")
    compiled_responses += f"# Response from competitor {index+1}\n\n"
    compiled_responses += answer + "\n\n"

logger.info(f"Compiled responses total length: {len(compiled_responses)} characters")

# Step 4: Judge using llama3.2:1b
logger.info("\n[Step 4] Judging responses using llama3.2:1b...\n")

judge_prompt = f"""You are judging a competition between {len(competitors)} competitors.
Each model has been given this question:

{question}

Your job is to evaluate each response for clarity and strength of argument, and rank them in order of best to worst.
Respond with JSON, and only JSON, with the following format (use proper JSON syntax with no trailing commas):
{{"results": [1, 2, 3]}}

The numbers represent the competitor numbers (1 for first competitor, 2 for second, etc.) in ranked order from best to worst.

Here are the responses from each competitor:

{compiled_responses}

Now respond ONLY with valid JSON showing the ranked order. Example: {{"results": [3, 1, 2]}} means competitor 3 is best, competitor 1 is second, competitor 2 is third.
Do not include markdown formatting, code blocks, or any explanation. Just the JSON."""

try:
    logger.info("Sending judge prompt to llama3.2:1b...")
    logger.info(f"Judge prompt length: {len(judge_prompt)} characters")

    judge_response = ollama_client.chat.completions.create(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": judge_prompt}]
    )
    judge_result = judge_response.choices[0].message.content

    logger.info(f"Raw judge response:\n{judge_result}\n")
    logger.info("="*80)

    # Step 5: Parse and display results
    logger.info("\n[Step 5] Final Rankings:\n")

    # Clean the response if it contains markdown code blocks
    cleaned_result = judge_result.strip()
    if cleaned_result.startswith("```"):
        logger.info("Cleaning markdown code blocks from judge response")
        # Remove markdown code block formatting
        lines = cleaned_result.split('\n')
        cleaned_result = '\n'.join([line for line in lines if not line.startswith("```")])

    logger.info(f"Cleaned judge result: {cleaned_result}")

    # Additional cleaning: remove trailing commas before closing brackets/braces
    cleaned_result = re.sub(r',(\s*[}\]])', r'\1', cleaned_result)
    logger.info(f"After comma cleanup: {cleaned_result}")

    results_dict = json.loads(cleaned_result)
    ranks = results_dict["results"]

    logger.info(f"Parsed rankings: {ranks}")

    for index, result in enumerate(ranks):
        competitor_index = int(result) - 1
        competitor = competitors[competitor_index]
        logger.info(f"🏆 Rank {index+1}: {competitor}")

    logger.info("\n" + "="*80)

except Exception as e:
    logger.error(f"Error during Judge #1 (llama3.2:1b) judging: {e}")
    logger.exception("Full traceback:")
    ranks = None

# Step 4b: Second Judge using gpt-4o-mini
logger.info("\n" + "="*80)
logger.info("\n[Step 4b] Getting second opinion from Judge #2 (gpt-4o-mini)...\n")

judge2_prompt = f"""You are an expert judge evaluating AI model responses in a competition.

{len(competitors)} AI models were asked this challenging question:

{question}

Here are their responses:

{compiled_responses}

Your task: Carefully analyze each response for:
1. Completeness - Did they address all parts of the question?
2. Depth of analysis - Quality of reasoning and detail
3. Practicality - Are the recommendations realistic and implementable?
4. Clarity - How well-structured and understandable is the response?

Provide your ranking as JSON with this exact format:
{{"results": [1, 2, 3, 4]}}

Where the numbers are competitor IDs (1=first, 2=second, etc.) ranked from best to worst.
Return ONLY the JSON, no other text."""

try:
    logger.info("Sending judge prompt to gpt-4o-mini...")

    judge2_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": judge2_prompt}],
        max_tokens=200,
        temperature=0.3
    )
    judge2_result = judge2_response.choices[0].message.content

    logger.info(f"Raw Judge #2 response:\n{judge2_result}\n")

    # Clean and parse
    cleaned_judge2 = judge2_result.strip()
    if cleaned_judge2.startswith("```"):
        lines = cleaned_judge2.split('\n')
        cleaned_judge2 = '\n'.join([line for line in lines if not line.startswith("```")])

    cleaned_judge2 = re.sub(r',(\s*[}\]])', r'\1', cleaned_judge2)
    logger.info(f"Cleaned Judge #2 result: {cleaned_judge2}")

    judge2_dict = json.loads(cleaned_judge2)
    ranks2 = judge2_dict["results"]

    logger.info(f"Judge #2 Rankings: {ranks2}")

    for index, result in enumerate(ranks2):
        competitor_index = int(result) - 1
        competitor = competitors[competitor_index]
        logger.info(f"🏆 Judge #2 Rank {index+1}: {competitor}")

except Exception as e:
    logger.error(f"Error during Judge #2 (gpt-4o-mini) judging: {e}")
    logger.exception("Full traceback:")
    ranks2 = None

# Display the actual answers
logger.info("\n" + "="*80)
logger.info("\n[DETAILED RESPONSES]\n")
for i, (competitor, answer) in enumerate(zip(competitors, answers)):
    logger.info(f"--- {competitor} ---")
    logger.info(answer)
    logger.info("\n")

logger.info("="*80)
logger.info("COMPETITION COMPLETE!")
logger.info("="*80)
logger.info(f"Full log saved to: {log_filename}")

# Generate HTML report
def generate_html_report(question, competitors, answers, judge_result=None, ranks=None, judge2_result=None, ranks2=None):
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Model Competition Results - {timestamp}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        .info-box {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .question {{
            background-color: #fff;
            padding: 20px;
            border-left: 4px solid #e74c3c;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .response {{
            background-color: #fff;
            padding: 20px;
            margin: 15px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .model-name {{
            font-weight: bold;
            color: #2980b9;
            font-size: 1.2em;
            margin-bottom: 10px;
        }}
        .rank {{
            display: inline-block;
            background-color: #f39c12;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-right: 10px;
        }}
        .rank-1 {{ background-color: #f1c40f; }}
        .rank-2 {{ background-color: #95a5a6; }}
        .rank-3 {{ background-color: #cd7f32; }}
        .stats {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: #3498db;
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            flex: 1;
            margin: 0 10px;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
        }}
        pre {{
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .refusal {{
            background-color: #ffe6e6;
            border-left: 4px solid #e74c3c;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>🏆 Multi-Model Competition Results</h1>
    <div class="info-box">
        <p class="timestamp"><strong>Run Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Log File:</strong> {log_filename}</p>
        <p><strong>HTML Report:</strong> {html_filename}</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-number">{len(models)}</div>
            <div>Models Tested</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{len(competitors)}</div>
            <div>Valid Responses</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{len(models) - len(competitors)}</div>
            <div>Refusals/Errors</div>
        </div>
    </div>

    <h2>📋 Challenge Question</h2>
    <div class="question">
        <pre>{question}</pre>
    </div>

    <h2>🤖 Model Responses</h2>
"""

    # Add responses with rankings if available
    for i, (competitor, answer) in enumerate(zip(competitors, answers), 1):
        refusal_check = any(phrase.lower() in answer.lower() for phrase in ["I can't help with this", "I cannot assist"])
        response_class = "response refusal" if refusal_check else "response"

        rank_badge = ""
        if ranks and i <= len(ranks):
            try:
                rank_position = ranks.index(i) + 1
                rank_class = f"rank-{rank_position}" if rank_position <= 3 else "rank"
                rank_badge = f'<span class="{rank_class}">Rank #{rank_position}</span>'
            except (ValueError, IndexError):
                pass

        html += f"""
    <div class="{response_class}">
        <div class="model-name">{rank_badge} {competitor}</div>
        <p><strong>Response Length:</strong> {len(answer)} characters</p>
        <pre>{answer}</pre>
    </div>
"""

    # Add judge verdicts if available
    if judge_result or judge2_result:
        html += """
    <h2>⚖️ Judges' Verdicts</h2>
"""

    if judge_result:
        html += f"""
    <h3>Judge #1: llama3.2:1b (Local Ollama)</h3>
    <div class="response">
        <pre>{judge_result}</pre>
    </div>
"""

    if judge2_result:
        html += f"""
    <h3>Judge #2: gpt-4o-mini (OpenAI)</h3>
    <div class="response">
        <pre>{judge2_result}</pre>
    </div>
"""

    # Add summary section
    html += """
    <h2>📊 Summary</h2>
    <div class="info-box">
"""

    # Show empty response models
    empty_models = [comp for comp, ans in zip(competitors, answers) if len(ans) == 0]
    if empty_models:
        html += f"        <p><strong>⚠️ Empty Responses:</strong> {', '.join(empty_models)}</p>\n"

    # Show successful models
    successful_models = [comp for comp, ans in zip(competitors, answers) if len(ans) > 0]
    if successful_models:
        html += f"        <p><strong>✅ Successful Responses:</strong> {', '.join(successful_models)}</p>\n"

    # Show rankings if available
    if ranks or ranks2:
        html += "        <p><strong>🏆 Rankings Comparison:</strong></p>\n"
        html += "        <table style='width:100%; border-collapse: collapse;'>\n"
        html += "            <tr style='background-color: #3498db; color: white;'>\n"
        html += "                <th style='padding: 10px; border: 1px solid #ddd;'>Rank</th>\n"

        if ranks:
            html += "                <th style='padding: 10px; border: 1px solid #ddd;'>Judge #1 (llama3.2:1b)</th>\n"
        if ranks2:
            html += "                <th style='padding: 10px; border: 1px solid #ddd;'>Judge #2 (gpt-4o-mini)</th>\n"

        html += "            </tr>\n"

        max_ranks = max(len(ranks) if ranks else 0, len(ranks2) if ranks2 else 0)

        for i in range(max_ranks):
            html += "            <tr>\n"
            html += f"                <td style='padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: bold;'>#{i+1}</td>\n"

            if ranks and i < len(ranks):
                try:
                    comp_idx = int(ranks[i]) - 1
                    if 0 <= comp_idx < len(competitors):
                        html += f"                <td style='padding: 10px; border: 1px solid #ddd;'>{competitors[comp_idx]}</td>\n"
                    else:
                        html += "                <td style='padding: 10px; border: 1px solid #ddd;'>-</td>\n"
                except:
                    html += "                <td style='padding: 10px; border: 1px solid #ddd;'>-</td>\n"
            elif ranks:
                html += "                <td style='padding: 10px; border: 1px solid #ddd;'>-</td>\n"

            if ranks2 and i < len(ranks2):
                try:
                    comp_idx = int(ranks2[i]) - 1
                    if 0 <= comp_idx < len(competitors):
                        html += f"                <td style='padding: 10px; border: 1px solid #ddd;'>{competitors[comp_idx]}</td>\n"
                    else:
                        html += "                <td style='padding: 10px; border: 1px solid #ddd;'>-</td>\n"
                except:
                    html += "                <td style='padding: 10px; border: 1px solid #ddd;'>-</td>\n"
            elif ranks2:
                html += "                <td style='padding: 10px; border: 1px solid #ddd;'>-</td>\n"

            html += "            </tr>\n"

        html += "        </table>\n"

    html += """
    </div>
</body>
</html>
"""
    return html

# Collect judge information if available
judge_info = None
judge2_info = None
ranking_info = None
ranking2_info = None

if 'judge_result' in locals():
    judge_info = judge_result
if 'judge2_result' in locals():
    judge2_info = judge2_result
if 'ranks' in locals():
    ranking_info = ranks
if 'ranks2' in locals():
    ranking2_info = ranks2

try:
    html_output = generate_html_report(question, competitors, answers, judge_info, ranking_info, judge2_info, ranking2_info)

    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_output)

    logger.info(f"HTML report saved to: {html_filename}")
    print(f"\n✅ HTML report generated: {html_filename}")
    print(f"✅ Log file saved: {log_filename}")
except Exception as e:
    logger.error(f"Error generating HTML report: {e}")
    logger.exception("Full traceback:")
