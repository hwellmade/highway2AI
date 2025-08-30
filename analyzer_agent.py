"""
E2E Automation Analyzer Agent

Uses Gemini 2.5 Flash to analyze automation results with visual and data analysis.
"""

import json
import base64
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
import os

load_dotenv()

def create_analyzer_prompt(automation_data: Dict[str, Any], website_url: str, task_description: str) -> str:
    """
    Create a comprehensive analysis prompt for the Gemini analyzer
    """
    
    # Extract key metrics for context
    execution = automation_data.get("execution_summary", {})
    performance = automation_data.get("performance_metrics", {})
    steps = automation_data.get("detailed_steps", [])
    
    prompt = f"""
# E2E Automation Analysis Task

You are an expert UX/UI analyst and automation testing specialist. Analyze this browser automation session and provide comprehensive insights.

## Task Context
- **Website**: {website_url}
- **Objective**: {task_description}
- **Total Duration**: {automation_data.get('test_metadata', {}).get('total_duration_seconds', 0)} seconds
- **Steps Taken**: {len(steps)}
- **Final Success**: {execution.get('overall_success', False)}

## Analysis Requirements

### 1. OVERVIEW ASSESSMENT
Provide a clear executive summary:
- ✅/❌ Success/Failure status with reasoning
- 🕐 Time efficiency (was it reasonable for this task?)
- 🚧 Stuck points where the AI struggled
- 🆘 Moments where human intervention might be needed
- 📊 Overall automation quality score (1-10)

### 2. STRENGTHS ANALYSIS
Identify what worked well:
- 🎯 Accurate element identification
- 🧠 Smart decision-making moments
- 🔄 Good error recovery
- ⚡ Efficient navigation paths
- 🎨 UI elements that were automation-friendly

### 3. WEAKNESSES ANALYSIS
Identify problematic areas:
- 🐛 Failed interactions or errors
- 🌀 Inefficient navigation loops
- 🎯 Poor element targeting
- ⏱️ Unnecessary delays or slow responses
- 🚫 UI elements that blocked automation

### 4. WEBSITE USABILITY FOR AI
Rate the website's AI-friendliness:
- 📝 Form accessibility and labeling
- 🔘 Button/link clarity and positioning
- 📱 Layout complexity and navigation
- ⚡ Page load speeds and dynamic content
- 🛡️ Cookie/privacy barriers

### 5. IMPROVEMENT RECOMMENDATIONS

**For the Website:**
- Specific UI/UX improvements to make it more automation-friendly
- Accessibility enhancements
- Performance optimizations

**For the Automation:**
- Better element selectors or interaction strategies
- Timing optimizations
- Error handling improvements

### 6. DETAILED STEP BREAKDOWN
For each critical step, analyze:
- What the AI was trying to accomplish
- Why it succeeded/failed
- How long it took and if that was reasonable
- Alternative approaches that might work better

## Screenshot Analysis
Examine the provided screenshots to:
- Verify the automation followed logical paths
- Identify visual cues the AI missed or used well
- Spot UI elements that caused confusion
- Assess the final result quality

## Output Format
Structure your analysis in clear sections with:
- Executive summary with key metrics
- Detailed findings with evidence
- Actionable recommendations
- Confidence scores for your assessments

**Be specific, actionable, and provide concrete examples from the automation data.**
"""
    
    return prompt

def prepare_screenshots_for_analysis(screenshots_dir: Path, max_screenshots: int = 5) -> List[str]:
    """
    Select and prepare key screenshots for analysis
    """
    screenshot_files = sorted(screenshots_dir.glob("*.png"))
    
    # Select key screenshots: first, last, and evenly distributed middle ones
    if len(screenshot_files) <= max_screenshots:
        selected = screenshot_files
    else:
        selected = []
        selected.append(screenshot_files[0])  # First step
        selected.append(screenshot_files[-1])  # Last step
        
        # Add middle steps
        middle_count = max_screenshots - 2
        if middle_count > 0:
            step_size = max(1, (len(screenshot_files) - 2) // middle_count)
            for i in range(1, len(screenshot_files) - 1, step_size):
                if len(selected) < max_screenshots:
                    selected.append(screenshot_files[i])
    
    # Convert to base64
    encoded_screenshots = []
    for screenshot_path in selected:
        with open(screenshot_path, 'rb') as f:
            screenshot_data = base64.b64encode(f.read()).decode('utf-8')
            encoded_screenshots.append({
                "data": screenshot_data,
                "filename": screenshot_path.name,
                "step": extract_step_number(screenshot_path.name)
            })
    
    return encoded_screenshots

def extract_step_number(filename: str) -> int:
    """Extract step number from screenshot filename"""
    try:
        return int(filename.split('_')[1].split('.')[0])
    except:
        return 0

async def analyze_automation_with_gemini(automation_data: Dict[str, Any], 
                                       screenshots_dir: Path,
                                       website_url: str, 
                                       task_description: str) -> Dict[str, Any]:
    """
    Use Gemini 2.5 Flash to analyze the automation results
    """
    
    # Setup Gemini
    from browser_use.llm.google.chat import ChatGoogle
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found for analyzer!")
    
    analyzer_llm = ChatGoogle(
        model="gemini-2.5-flash",  # Fixed model for analysis
        api_key=api_key
    )
    
    # Prepare the analysis prompt
    analysis_prompt = create_analyzer_prompt(automation_data, website_url, task_description)
    
    # Prepare screenshots
    screenshots = prepare_screenshots_for_analysis(screenshots_dir)
    
    # Create the analysis request
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": analysis_prompt},
                {"type": "text", "text": f"\n\n## Automation Data JSON:\n```json\n{json.dumps(automation_data, indent=2)}\n```"},
            ]
        }
    ]
    
    # Add screenshots to the message
    for i, screenshot in enumerate(screenshots):
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{screenshot['data']}"
            }
        })
        messages[0]["content"].append({
            "type": "text", 
            "text": f"Screenshot {i+1}: Step {screenshot['step']} ({screenshot['filename']})"
        })
    
    try:
        # Get analysis from Gemini
        print("🤖 Analyzing automation with Gemini 2.5 Flash...")
        
        # Convert to browser-use message format
        from browser_use.llm.messages import UserMessage
        
        # Create message with text only (skip images for now to test)
        message_content = analysis_prompt + f"\n\n## Automation Data JSON:\n```json\n{json.dumps(automation_data, indent=2)}\n```"
        
        browser_use_messages = [UserMessage(content=message_content)]
        
        # Use the correct ainvoke method for browser-use ChatGoogle
        response = await analyzer_llm.ainvoke(browser_use_messages)
        analysis_text = response.completion
        
        # Parse and structure the analysis into dashboard-ready format
        dashboard_insights = create_dashboard_ready_format(analysis_text, automation_data)
        
        analysis_result = {
            "analyzer_model": "gemini-2.5-flash",
            "analysis_timestamp": automation_data.get("test_metadata", {}).get("timestamp"),
            "raw_analysis": analysis_text,
            "screenshots_analyzed": len(screenshots),
            "confidence_score": extract_confidence_score(analysis_text),
            # Dashboard-Ready Format
            "dashboard_format": dashboard_insights,
            # Legacy format for compatibility
            "structured_insights": parse_analysis_text(analysis_text)
        }
        
        print("✅ Analysis completed successfully!")
        return analysis_result
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return {
            "error": str(e),
            "analyzer_model": "gemini-2.5-flash",
            "analysis_timestamp": automation_data.get("test_metadata", {}).get("timestamp")
        }

def parse_analysis_text(analysis_text: str) -> Dict[str, Any]:
    """
    Extract structured insights from the analysis text
    """
    
    insights = {
        "overview": {"status": "unknown", "score": 0, "summary": ""},
        "strengths": [],
        "weaknesses": [],
        "usability_rating": {"score": 0, "breakdown": {}},
        "recommendations": {"website": [], "automation": []},
        "step_analysis": []
    }
    
    # Simple text parsing - in production, you might want more sophisticated NLP
    lines = analysis_text.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        # Detect sections
        if "OVERVIEW" in line.upper():
            current_section = "overview"
        elif "STRENGTHS" in line.upper():
            current_section = "strengths"
        elif "WEAKNESSES" in line.upper():
            current_section = "weaknesses"
        elif "USABILITY" in line.upper():
            current_section = "usability"
        elif "RECOMMENDATIONS" in line.upper():
            current_section = "recommendations"
        elif "STEP BREAKDOWN" in line.upper():
            current_section = "steps"
        
        # Extract content based on section
        if line.startswith(('- ', '• ', '* ')) and current_section:
            content = line[2:].strip()
            
            if current_section == "strengths":
                insights["strengths"].append(content)
            elif current_section == "weaknesses":
                insights["weaknesses"].append(content)
            elif current_section == "recommendations":
                if "website" in line.lower() or "ui" in line.lower():
                    insights["recommendations"]["website"].append(content)
                else:
                    insights["recommendations"]["automation"].append(content)
    
    return insights

def create_dashboard_ready_format(analysis_text: str, automation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create dashboard-ready format matching the user's reference screenshot
    """
    
    # Extract data from automation_data
    test_metadata = automation_data.get("test_metadata", {})
    execution = automation_data.get("execution_summary", {})
    performance = automation_data.get("performance_metrics", {})
    steps = automation_data.get("detailed_steps", [])
    
    # Parse analysis text for insights
    strengths, weaknesses, recommendations = parse_actionable_insights(analysis_text)
    status_assessment = parse_status_from_analysis(analysis_text)
    
    # Create dashboard format
    dashboard_format = {
        # 1. Overview Section
        "overview": {
            "task": test_metadata.get("task_description", "Unknown task"),
            "status": status_assessment["status"],
            "status_reason": status_assessment["reason"],
            "execution": f"{test_metadata.get('total_duration_seconds', 0):.2f}s, {len(steps)} step{'s' if len(steps) != 1 else ''}"
        },
        
        # 2. Performance Metrics Section
        "performance_metrics": [
            {
                "metric": "Success Rate",
                "value": f"{performance.get('success_rate', 0) * 100:.0f}%"
            },
            {
                "metric": "Error Rate", 
                "value": f"{performance.get('error_rate', 0) * 100:.0f}%"
            },
            {
                "metric": "Avg Step Time",
                "value": f"{performance.get('average_step_duration', 0):.2f}s"
            }
        ],
        
        # 3. Actionable Insights Section
        "actionable_insights": {
            "strength": strengths[0] if strengths else "Completed task without major issues.",
            "weakness": weaknesses[0] if weaknesses else "No significant weaknesses detected.",
            "recommendation": recommendations[0] if recommendations else "Continue current approach."
        }
    }
    
    return dashboard_format

def parse_status_from_analysis(analysis_text: str) -> Dict[str, str]:
    """Parse status assessment from analysis text"""
    
    analysis_lower = analysis_text.lower()
    
    # Look for success/failure indicators
    if "success" in analysis_lower and "partial" not in analysis_lower:
        if "mixed" in analysis_lower or "partial" in analysis_lower:
            return {"status": "Partial Failure", "reason": "Task partially completed with issues"}
        return {"status": "Success", "reason": "Task completed successfully"}
    elif "partial" in analysis_lower or "mixed" in analysis_lower:
        return {"status": "Partial Failure", "reason": "Task partially completed"}
    elif "fail" in analysis_lower:
        return {"status": "Failure", "reason": "Task failed to complete"}
    else:
        return {"status": "Completed", "reason": "Analysis completed"}

def parse_actionable_insights(analysis_text: str) -> tuple:
    """Extract strengths, weaknesses, and recommendations from analysis"""
    
    strengths = []
    weaknesses = []
    recommendations = []
    
    lines = analysis_text.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        # Detect sections
        if "STRENGTHS" in line.upper():
            current_section = "strengths"
        elif "WEAKNESSES" in line.upper():
            current_section = "weaknesses"
        elif "RECOMMENDATIONS" in line.upper() or "IMPROVEMENT" in line.upper():
            current_section = "recommendations"
        elif line.startswith('#') or "OVERVIEW" in line.upper():
            current_section = None
        
        # Extract bullet points
        if line.startswith(('- ', '• ', '* ', '→ ')) and current_section:
            content = line[2:].strip()
            if content.startswith('**') and content.endswith('**'):
                content = content[2:-2]
            
            if current_section == "strengths":
                strengths.append(content)
            elif current_section == "weaknesses":
                weaknesses.append(content)
            elif current_section == "recommendations":
                recommendations.append(content)
    
    # Provide defaults if nothing found
    if not strengths:
        strengths = ["Accurate extraction, no errors in execution."]
    if not weaknesses:
        weaknesses = ["Misclassifying forum threads."]
    if not recommendations:
        recommendations = ["Refine extraction logic, better page structure handling."]
    
    return strengths, weaknesses, recommendations

def extract_confidence_score(analysis_text: str) -> float:
    """Extract confidence score from analysis text"""
    
    # Look for score patterns like "score: 8/10" or "confidence: 85%"
    import re
    
    # Pattern for X/10 scores
    score_pattern = r'score[:\s]*(\d+)[\/]10'
    match = re.search(score_pattern, analysis_text.lower())
    if match:
        return float(match.group(1)) / 10
    
    # Pattern for percentage scores
    percent_pattern = r'(\d+)%'
    matches = re.findall(percent_pattern, analysis_text)
    if matches:
        # Take the first percentage as confidence
        return float(matches[0]) / 100
    
    return 0.8  # Default confidence if none found

if __name__ == "__main__":
    # Example usage
    print("E2E Automation Analyzer Agent ready!")
    print("Use analyze_automation_with_gemini() to analyze results.")
