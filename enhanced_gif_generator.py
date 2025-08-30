"""
Enhanced GIF generator that adds AI thinking process to each frame.

This creates GIFs with:
- Step number
- AI thinking process
- Goal/objective
- Action taken
- Screenshot of the step
"""

import base64
import io
import logging
from pathlib import Path
from typing import List

from browser_use.agent.views import AgentHistoryList

# Import PIL components
try:
    from PIL import Image, ImageFont, ImageDraw
except ImportError:
    Image = None
    ImageFont = None
    ImageDraw = None

logger = logging.getLogger(__name__)


def create_enhanced_gif_with_thinking(
    task: str,
    history: AgentHistoryList,
    output_path: str = 'enhanced_automation.gif',
    duration: int = 4000,  # Longer duration to read thinking
    font_size: int = 16,   # Smaller font for more text
    margin: int = 20,
    max_thinking_chars: int = 200,  # Limit thinking text length
) -> None:
    """Create a GIF with AI thinking process overlaid on each frame."""
    
    if not history.history:
        logger.warning('No history to create GIF from')
        return

    if Image is None:
        logger.error("PIL/Pillow not available for GIF generation")
        return

    images = []
    
    # Get all screenshots from history
    screenshots = history.screenshots(return_none_if_not_screenshot=True)
    
    if not screenshots:
        logger.warning('No screenshots found in history')
        return

    # Try to load a good font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        title_font = ImageFont.truetype("arial.ttf", font_size + 4)
    except (OSError, IOError):
        try:
            # Try system fonts on Windows
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
            title_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size + 4)
        except (OSError, IOError):
            # Fallback to default font
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()

    # Create task introduction frame
    if screenshots:
        first_screenshot = None
        for screenshot in screenshots:
            if screenshot:
                first_screenshot = screenshot
                break
        
        if first_screenshot:
            task_frame = _create_task_introduction_frame(
                task, first_screenshot, title_font, font
            )
            images.append(task_frame)

    # Process each history item with enhanced overlay
    for i, (item, screenshot) in enumerate(zip(history.history, screenshots), 1):
        if not screenshot:
            continue

        # Convert base64 screenshot to PIL Image
        try:
            img_data = base64.b64decode(screenshot)
            image = Image.open(io.BytesIO(img_data))
        except Exception as e:
            logger.warning(f"Could not process screenshot for step {i}: {e}")
            continue

        # Add enhanced overlay with thinking
        if item.model_output:
            image = _add_enhanced_overlay(
                image=image,
                step_number=i,
                thinking_text=item.model_output.thinking or "",
                goal_text=item.model_output.current_state.next_goal or "",
                actions=item.model_output.action,
                font=font,
                title_font=title_font,
                margin=margin,
                max_thinking_chars=max_thinking_chars
            )

        images.append(image)

    if images:
        # Save the enhanced GIF
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
            optimize=False,
        )
        logger.info(f'Created enhanced GIF at {output_path}')
    else:
        logger.warning('No images found to create enhanced GIF')


def _create_task_introduction_frame(task: str, first_screenshot: str, title_font, font):
    """Create an introduction frame explaining the task."""
    
    img_data = base64.b64decode(first_screenshot)
    template = Image.open(io.BytesIO(img_data))
    
    # Create a dark overlay for better text readability
    image = template.copy()
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 180))  # Semi-transparent black
    image = Image.alpha_composite(image.convert('RGBA'), overlay)
    
    draw = ImageDraw.Draw(image)
    
    # Calculate center position
    center_y = image.height // 2
    
    # Title text
    title = "🤖 AI Agent E2E Test"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (image.width - title_width) // 2
    title_y = center_y - 60
    
    draw.text((title_x, title_y), title, font=title_font, fill=(255, 255, 255))
    
    # Task description
    task_lines = _wrap_text(f"Task: {task}", font, image.width - 100)
    lines = task_lines.split('\n')
    
    line_height = font.size + 5
    total_text_height = len(lines) * line_height
    start_y = center_y - 20
    
    for i, line in enumerate(lines):
        line_bbox = draw.textbbox((0, 0), line, font=font)
        line_width = line_bbox[2] - line_bbox[0]
        line_x = (image.width - line_width) // 2
        line_y = start_y + (i * line_height)
        
        draw.text((line_x, line_y), line, font=font, fill=(255, 255, 255))
    
    return image.convert('RGB')


def _add_enhanced_overlay(
    image,
    step_number: int,
    thinking_text: str,
    goal_text: str,
    actions: List,
    font,
    title_font,
    margin: int,
    max_thinking_chars: int
):
    """Add enhanced overlay with thinking process, goals, and actions."""
    
    image = image.convert('RGBA')
    
    # Create a text panel on the right side
    panel_width = 350
    panel_height = image.height
    
    # Create the main image with extra width for text panel
    new_width = image.width + panel_width
    enhanced_image = Image.new('RGB', (new_width, image.height), (240, 240, 240))
    
    # Paste the original screenshot on the left
    enhanced_image.paste(image, (0, 0))
    
    # Create the text panel
    draw = ImageDraw.Draw(enhanced_image)
    
    # Panel background
    panel_x = image.width
    draw.rectangle(
        [(panel_x, 0), (new_width, panel_height)],
        fill=(45, 45, 45),  # Dark gray background
        outline=(70, 70, 70),
        width=2
    )
    
    # Text positioning
    text_x = panel_x + 15
    current_y = 15
    line_height = font.size + 3
    
    # Step number header
    step_text = f"🔍 Step {step_number}"
    draw.text((text_x, current_y), step_text, font=title_font, fill=(100, 200, 255))
    current_y += title_font.size + 10
    
    # Thinking section
    if thinking_text:
        # Truncate thinking if too long
        if len(thinking_text) > max_thinking_chars:
            thinking_text = thinking_text[:max_thinking_chars] + "..."
        
        draw.text((text_x, current_y), "🧠 AI Thinking:", font=title_font, fill=(255, 200, 100))
        current_y += title_font.size + 5
        
        thinking_lines = _wrap_text(thinking_text, font, panel_width - 30)
        for line in thinking_lines.split('\n'):
            if current_y < panel_height - 20:  # Don't overflow
                draw.text((text_x, current_y), line, font=font, fill=(255, 255, 255))
                current_y += line_height
        
        current_y += 10
    
    # Goal section
    if goal_text and current_y < panel_height - 60:
        draw.text((text_x, current_y), "🎯 Goal:", font=title_font, fill=(100, 255, 100))
        current_y += title_font.size + 5
        
        goal_lines = _wrap_text(goal_text, font, panel_width - 30)
        for line in goal_lines.split('\n'):
            if current_y < panel_height - 20:
                draw.text((text_x, current_y), line, font=font, fill=(200, 255, 200))
                current_y += line_height
        
        current_y += 10
    
    # Actions section
    if actions and current_y < panel_height - 40:
        draw.text((text_x, current_y), "⚡ Actions:", font=title_font, fill=(255, 150, 150))
        current_y += title_font.size + 5
        
        for action in actions[:2]:  # Limit to first 2 actions to save space
            if current_y < panel_height - 20:
                action_text = f"• {action.__class__.__name__}"
                draw.text((text_x, current_y), action_text, font=font, fill=(255, 200, 200))
                current_y += line_height
    
    return enhanced_image


def _wrap_text(text: str, font, max_width: int) -> str:
    """Wrap text to fit within a given width."""
    
    try:
        # Create a temporary draw object to measure text
        temp_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_img)
        
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            line = ' '.join(current_line)
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            
            if line_width > max_width:
                if len(current_line) == 1:
                    lines.append(current_line.pop())
                else:
                    current_line.pop()
                    lines.append(' '.join(current_line))
                    current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    except Exception:
        # Fallback: simple character-based wrapping
        max_chars_per_line = max_width // 8  # Rough estimate
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= max_chars_per_line:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return '\n'.join(lines)


# Integration function for the main solution
def generate_enhanced_automation_gif(history: AgentHistoryList, task: str, output_path: str):
    """Generate enhanced GIF with AI thinking for the main solution."""
    
    print("🎬 Generating enhanced GIF with AI thinking process...")
    
    try:
        create_enhanced_gif_with_thinking(
            task=task,
            history=history,
            output_path=output_path,
            duration=5000,  # 5 seconds per frame for reading
            max_thinking_chars=250,
        )
        
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size / (1024*1024)
            print(f"✅ Enhanced GIF created: {output_path} ({file_size:.1f} MB)")
            return True
        else:
            print("❌ Enhanced GIF creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced GIF generation error: {e}")
        return False
