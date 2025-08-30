"""
E2E Automation Analyzer - FastAPI Web Application

A web interface for analyzing website automation performance using browser-use
and AI-powered analysis. Built for hackathon demo.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, Any
import uuid
import base64

from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

# Browser-use imports
from browser_use import Agent
from browser_use.agent.views import AgentHistoryList
from browser_use.browser import BrowserSession, BrowserProfile
from browser_use.browser.profile import ViewportSize
from enhanced_gif_generator import generate_enhanced_automation_gif

load_dotenv()

# FastAPI app setup
app = FastAPI(
    title="E2E Automation Analyzer",
    description="Analyze website automation performance with AI agents",
    version="1.0.0"
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory="e2e_analysis_results"), name="results")
templates = Jinja2Templates(directory="templates")

# In-memory job storage (for demo - use Redis/DB in production)
jobs: Dict[str, Dict[str, Any]] = {}

# Pydantic models
class AnalysisRequest(BaseModel):
    website_url: str
    task_description: str
    llm_model: str = "gpt-4o-mini"
    max_steps: int = 10

class AnalysisStatus(BaseModel):
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: str
    result: Dict[str, Any] = None
    error: str = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main page with analysis form"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def start_analysis(
    background_tasks: BackgroundTasks,
    website_url: str = Form(...),
    task_description: str = Form(...),
    llm_model: str = Form(default="gpt-4o-mini"),
    max_steps: int = Form(default=10)
):
    """Start automation analysis in background"""
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    jobs[job_id] = {
        "status": "pending",
        "progress": "Initializing automation analysis...",
        "website_url": website_url,
        "task_description": task_description,
        "llm_model": llm_model,
        "max_steps": max_steps,
        "created_at": time.time()
    }
    
    # Start background task
    background_tasks.add_task(
        run_analysis_background, 
        job_id, 
        website_url, 
        task_description, 
        llm_model, 
        max_steps
    )
    
    return JSONResponse({
        "job_id": job_id,
        "status": "started",
        "message": "Analysis started successfully"
    })

@app.get("/status/{job_id}")
async def get_analysis_status(job_id: str):
    """Get analysis status and results"""
    
    if job_id not in jobs:
        return JSONResponse(
            {"error": "Job not found"}, 
            status_code=404
        )
    
    job = jobs[job_id]
    
    return JSONResponse({
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "result": job.get("result"),
        "error": job.get("error"),
        "created_at": job["created_at"]
    })

@app.get("/results/{job_id}", response_class=HTMLResponse)
async def view_results(request: Request, job_id: str):
    """Display analysis results in web interface"""
    
    if job_id not in jobs:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "Analysis not found"
        })
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        return templates.TemplateResponse("status.html", {
            "request": request,
            "job": job,
            "job_id": job_id
        })
    
    return templates.TemplateResponse("results.html", {
        "request": request,
        "job": job,
        "job_id": job_id,
        "result": job.get("result", {})
    })

def setup_llm(llm_model: str = None):
    """Setup LLM with configurable provider and model from .env"""
    
    # Get provider and model from environment variables
    provider = os.getenv('LLM_PROVIDER', 'openrouter').lower()
    
    if provider == 'google':
        model = llm_model or os.getenv('GOOGLE_MODEL', 'gemini-2.5-flash')
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file!")
        
        from browser_use.llm.google.chat import ChatGoogle
        print(f"🚀 Using Google Gemini API - Model: {model}")
        return ChatGoogle(
            model=model,
            api_key=api_key
        )
    
    elif provider == 'openrouter':
        model = llm_model or os.getenv('OPENROUTER_MODEL', 'openai/gpt-4o-mini')
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file!")
        
        from browser_use.llm.openrouter.chat import ChatOpenRouter
        print(f"🔗 Using OpenRouter API - Model: {model}")
        return ChatOpenRouter(
            model=model,
            api_key=api_key
        )
    
    elif provider == 'openai':
        model = llm_model or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file!")
        
        from browser_use.llm.openai.chat import ChatOpenAI
        print(f"🔗 Using OpenAI API - Model: {model}")
        return ChatOpenAI(
            model=model,
            api_key=api_key
        )
    
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Supported: google, openrouter, openai")


async def run_automation_analysis(website_url: str, task_description: str, max_steps: int = 8):
    """
    Run automation analysis and return complete data for analyzer LLM.
    
    This is the core function that runs the browser automation.
    """
    
    print(f"🚀 Starting E2E automation analysis...")
    print(f"   🌐 Website: {website_url}")
    print(f"   📋 Task: {task_description}")
    
    # Create timestamped output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path("./e2e_analysis_results")
    output_dir = base_output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Results will be saved to: {output_dir}")
    
    # Setup components
    llm = setup_llm()
    
    # Create browser session (optimized for analysis)
    browser_session = BrowserSession(
        browser_profile=BrowserProfile(
            headless=False,  # Visible for debugging; set True for production
            window_size=ViewportSize(width=1280, height=800),
            user_data_dir=str(output_dir / "browser_data"),
            # Optimizations for analysis
            wait_between_actions=0.5,  # Small delay for better observation
            highlight_elements=True,   # Highlight interacted elements
        )
    )
    
    await browser_session.start()
    
    try:
        # Create agent (we'll generate enhanced GIF separately)
        agent = Agent(
            task=f"Go to {website_url} and {task_description}",
            llm=llm,
            browser_session=browser_session,
            # Don't use built-in GIF - we'll create enhanced version
        )
        
        print("🔄 Running automation...")
        start_time = time.time()
        
        # Run automation
        history: AgentHistoryList = await agent.run(max_steps=max_steps)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        print("✅ Automation completed!")
        
        # Generate enhanced GIF with AI thinking
        enhanced_gif_path = output_dir / "enhanced_automation_with_thinking.gif"
        gif_success = generate_enhanced_automation_gif(
            history=history,
            task=f"Go to {website_url} and {task_description}",
            output_path=str(enhanced_gif_path)
        )
        
        # Extract comprehensive data
        analysis_data = extract_analysis_data(
            history, website_url, task_description, total_duration, output_dir, gif_success, timestamp
        )
        
        return analysis_data
        
    except Exception as e:
        print(f"❌ Automation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "website_url": website_url,
            "task_description": task_description
        }
        
    finally:
        await browser_session.stop()


def extract_analysis_data(history: AgentHistoryList, website_url: str, task_description: str, 
                         total_duration: float, output_dir: Path, enhanced_gif_created: bool = False, timestamp: str = None):
    """Extract comprehensive data for analyzer LLM"""
    
    print("📊 Extracting analysis data...")
    
    # Basic results
    final_result = history.final_result()
    errors = history.errors()
    screenshots = history.screenshots()
    
    # Calculate metrics
    success_rate = 1.0 if final_result else 0.0
    error_rate = len(errors) / len(history.history) if history.history else 0.0
    avg_step_duration = total_duration / len(history.history) if history.history else 0.0
    
    # Extract detailed step analysis
    steps_analysis = []
    for i, step in enumerate(history.history, 1):
        step_analysis = {
            "step_number": i,
            "browser_context": {
                "url": step.state.url,
                "title": step.state.title,
                "has_screenshot": bool(step.state.get_screenshot()),
                "interactive_elements_count": len(step.state.interacted_element) if step.state.interacted_element else 0
            },
            "ai_decision_making": {
                "thinking_process": step.model_output.thinking if step.model_output else None,
                "goal_setting": step.model_output.current_state.next_goal if step.model_output else None,
                "success_evaluation": step.model_output.current_state.evaluation_previous_goal if step.model_output else None,
                "memory_state": step.model_output.current_state.memory if step.model_output else None
            },
            "actions_executed": [],
            "action_outcomes": [],
            "usability_indicators": {
                "successful_actions": 0,
                "failed_actions": 0,
                "retry_attempts": 0
            }
        }
        
        # Analyze actions
        if step.model_output:
            for action in step.model_output.action:
                action_info = {
                    "type": action.__class__.__name__,
                    "parameters": action.model_dump(),
                    "complexity_score": len(action.model_dump())  # Simple complexity metric
                }
                step_analysis["actions_executed"].append(action_info)
        
        # Analyze outcomes
        for result in step.result:
            outcome = result.model_dump()
            step_analysis["action_outcomes"].append(outcome)
            
            # Update usability indicators
            if result.error:
                step_analysis["usability_indicators"]["failed_actions"] += 1
            else:
                step_analysis["usability_indicators"]["successful_actions"] += 1
        
        steps_analysis.append(step_analysis)
    
    # Create comprehensive analysis data
    analysis_data = {
        "test_metadata": {
            "website_url": website_url,
            "task_description": task_description,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration_seconds": round(total_duration, 2),
            "browser_use_version": "0.7.0"
        },
        "execution_summary": {
            "overall_success": bool(final_result),
            "final_result": final_result,
            "total_steps": len(history.history),
            "total_errors": len(errors),
            "screenshots_captured": len(screenshots)
        },
        "performance_metrics": {
            "success_rate": success_rate,
            "error_rate": error_rate,
            "average_step_duration": round(avg_step_duration, 2),
            "efficiency_score": round((1 - error_rate) * success_rate, 2),
            "steps_per_minute": round(len(history.history) / (total_duration / 60), 1) if total_duration > 0 else 0
        },
        "usability_analysis": {
            "navigation_complexity": len(history.history),
            "interaction_success_rate": round(success_rate, 2),
            "error_patterns": [{"error": error, "frequency": 1} for error in errors],
            "ai_confidence_indicators": {
                "clear_goals": sum(1 for step in steps_analysis if step["ai_decision_making"]["goal_setting"]),
                "successful_evaluations": sum(1 for step in steps_analysis 
                                            if step["ai_decision_making"]["success_evaluation"] and 
                                            "success" in str(step["ai_decision_making"]["success_evaluation"]).lower())
            }
        },
        "detailed_steps": steps_analysis,
        "resource_usage": {
            "token_usage": history.usage.model_dump() if history.usage else None,
            "total_tokens": history.usage.total_tokens if history.usage else 0
        },
        "visual_documentation": {
            "enhanced_gif_with_thinking": f"/results/{timestamp}/enhanced_automation_with_thinking.gif" if enhanced_gif_created and timestamp else None,
            "gif_file": f"/results/{timestamp}/automation_flow.gif" if timestamp else str(output_dir / "automation_flow.gif"),
            "screenshots_directory": f"/results/{timestamp}/screenshots" if timestamp else str(output_dir / "screenshots"),
            "individual_screenshots": len(screenshots),
            "enhanced_gif_created": enhanced_gif_created,
            "timestamp": timestamp,
            "local_output_dir": str(output_dir)
        }
    }
    
    # Save analysis data
    analysis_file = output_dir / "e2e_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    # Save individual screenshots
    save_screenshots(screenshots, output_dir)
    
    print(f"💾 Analysis data saved to: {analysis_file}")
    return analysis_data


def save_screenshots(screenshots, output_dir: Path):
    """Save individual screenshots"""
    
    screenshots_dir = output_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)
    
    for i, screenshot_b64 in enumerate(screenshots, 1):
        if screenshot_b64:
            screenshot_file = screenshots_dir / f"step_{i}.png"
            screenshot_data = base64.b64decode(screenshot_b64)
            with open(screenshot_file, 'wb') as f:
                f.write(screenshot_data)
    
    print(f"📸 {len(screenshots)} screenshots saved to: {screenshots_dir}")


async def run_analysis_background(
    job_id: str, 
    website_url: str, 
    task_description: str, 
    llm_model: str, 
    max_steps: int
):
    """Background task to run automation analysis"""
    
    try:
        # Update status
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = "Setting up browser automation..."
        
        jobs[job_id]["progress"] = "Running browser automation..."
        
        # Run the automation analysis
        analysis_data = await run_automation_analysis(
            website_url=website_url,
            task_description=task_description,
            max_steps=max_steps
        )
        
        jobs[job_id]["progress"] = "Analyzing results with AI..."
        
        # Add analyzer LLM processing here (Phase 2)
        enhanced_analysis = await analyze_automation_results(analysis_data)
        
        # Update job with results
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = "Analysis completed successfully!"
        jobs[job_id]["result"] = enhanced_analysis
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["progress"] = f"Analysis failed: {str(e)}"

async def analyze_automation_results(automation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze automation results with second LLM to generate insights
    This is where we'll add the analyzer LLM in the next step
    """
    
    # For now, return the original data with some basic analysis
    # We'll enhance this with LLM analysis next
    
    enhanced_data = automation_data.copy()
    
    # Add some basic computed insights
    if "performance_metrics" in automation_data:
        metrics = automation_data["performance_metrics"]
        
        # Calculate overall grade
        success_weight = 0.4
        efficiency_weight = 0.3
        error_weight = 0.3
        
        overall_grade = (
            metrics.get("success_rate", 0) * success_weight +
            (1 - metrics.get("error_rate", 0)) * error_weight +
            metrics.get("efficiency_score", 0) * efficiency_weight
        ) * 100
        
        enhanced_data["ai_insights"] = {
            "overall_grade": round(overall_grade, 1),
            "grade_letter": get_grade_letter(overall_grade),
            "key_findings": generate_key_findings(automation_data),
            "recommendations": generate_recommendations(automation_data)
        }
    
    return enhanced_data

def get_grade_letter(score: float) -> str:
    """Convert numeric score to letter grade"""
    if score >= 90: return "A"
    elif score >= 80: return "B"
    elif score >= 70: return "C"
    elif score >= 60: return "D"
    else: return "F"

def generate_key_findings(data: Dict[str, Any]) -> list:
    """Generate key findings from automation data"""
    findings = []
    
    if "execution_summary" in data:
        summary = data["execution_summary"]
        if summary.get("overall_success"):
            findings.append("✅ Task completed successfully")
        else:
            findings.append("❌ Task failed to complete")
        
        findings.append(f"📊 Completed in {summary.get('total_steps', 0)} steps")
        
        if summary.get("total_errors", 0) > 0:
            findings.append(f"⚠️ Encountered {summary['total_errors']} errors")
    
    if "performance_metrics" in data:
        metrics = data["performance_metrics"]
        if metrics.get("success_rate", 0) == 1.0:
            findings.append("🎯 Perfect success rate")
        elif metrics.get("error_rate", 0) > 0.2:
            findings.append("🔍 High error rate detected")
    
    return findings

def generate_recommendations(data: Dict[str, Any]) -> list:
    """Generate recommendations based on automation data"""
    recommendations = []
    
    if "performance_metrics" in data:
        metrics = data["performance_metrics"]
        
        if metrics.get("error_rate", 0) > 0.1:
            recommendations.append("Consider improving element selectors and page load handling")
        
        if metrics.get("steps_per_minute", 0) < 5:
            recommendations.append("Automation could be optimized for faster execution")
        
        if len(data.get("detailed_steps", [])) > 15:
            recommendations.append("Website navigation could be simplified for better user experience")
    
    if not recommendations:
        recommendations.append("Automation performed well - no major issues detected")
    
    return recommendations

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
