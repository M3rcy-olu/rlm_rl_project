#!/usr/bin/env python3
import argparse
import re
import sys
import os
import html

# Add logtree to path
LOGTREE_PATH = "/Users/christophermejia/dev/dl_fp/smol_rlm/tinker-cookbook"
if LOGTREE_PATH not in sys.path:
    sys.path.append(LOGTREE_PATH)

try:
    import tinker_cookbook.utils.logtree as logtree
    from tinker_cookbook.utils.logtree_formatters import ConversationFormatter
except ImportError as e:
    # Fallback for relative path
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), "../smol_rlm/tinker-cookbook"))
        import tinker_cookbook.utils.logtree as logtree
        from tinker_cookbook.utils.logtree_formatters import ConversationFormatter
    except ImportError as e2:
        print(f"Could not import logtree. \nAttempt 1: {e}\nAttempt 2: {e2}")
        sys.exit(1)

def format_content_html(text):
    """
    Format text for HTML display:
    1. HTML escape everything first.
    2. Apply custom styling to code blocks and errors.
    """
    text = html.escape(text)
    
    lines = text.split('\n')
    formatted_lines = []
    
    in_code_block = False
    
    COLOR_CODE_BLOCK = "#b45309" # amber-700
    COLOR_ERROR = "#dc2626"      # red-600
    
    for line in lines:
        # Check for code blocks (```)
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            formatted_lines.append(f'<span style="color: {COLOR_CODE_BLOCK}; font-weight: bold;">{line}</span>')
            continue
            
        if in_code_block:
             formatted_lines.append(f'<span style="color: {COLOR_CODE_BLOCK};">{line}</span>')
        else:
            # Highlight errors
            if any(k in line for k in ["Error", "Exception", "Traceback", "ERROR"]):
                 formatted_lines.append(f'<span style="color: {COLOR_ERROR}; font-weight: bold;">{line}</span>')
            else:
                 formatted_lines.append(line)
                 
    return '\n'.join(formatted_lines)

def process_chunk(chunk_text, index):
    """
    Process a single chunk of text (separated by '---- datum ----').
    Logs it into the current logtree trace.
    """
    if not chunk_text.strip():
        return

    # Determine a title for this section
    # Look for trajectory header
    traj_match = re.search(r'\*+\s+trajectory\s+idx=(\d+)(?:,\s+reward=([0-9.-]+))?', chunk_text)
    if traj_match:
        title = f"Trajectory {traj_match.group(1)}"
        if traj_match.group(2):
            title += f" (Reward: {traj_match.group(2)})"
    else:
        title = f"Segment {index}"

    with logtree.scope_header(title):
        # 1. Trajectory/Meta Info extraction if not already in title (or to show details)
        # We can just show the header line if found
        if traj_match:
            logtree.log_text(traj_match.group(0), div_class="lt-subtitle")

        # 2. Per-step metrics
        metrics_match = re.search(r'Per-step metrics:(.*?)($|<\|im_start\|>)', chunk_text, re.DOTALL)
        if metrics_match:
            # Use logtree.details instead of scope_details(pre=True)
            logtree.details(metrics_match.group(1).strip(), summary="Per-step Metrics", pre=True)

        # 3. Conversation
        pattern = re.compile(r'<\|im_start\|>(.*?)\n(.*?)<\|im_end\|>', re.DOTALL)
        matches = list(pattern.finditer(chunk_text))
        
        if matches:
            messages = []
            for match in matches:
                role = match.group(1).strip()
                raw_content = match.group(2).strip()
                
                formatted_content = format_content_html(raw_content)
                
                messages.append({
                    "role": role,
                    "content": formatted_content
                })
            
            logtree.log_formatter(ConversationFormatter(messages))
        else:
            # If no conversation, checking if there is other content worth showing
            # Avoid showing the whole chunk if it was just headers we already parsed, 
            # but if there is unprocessed text, show it.
            # For simplicity, if no conversation, just show raw chunk in details
            cleaned_chunk = chunk_text.strip()
            # Remove the metrics part if we already showed it to avoid duplication? 
            # Nah, showing raw is fine for debug.
            logtree.details(cleaned_chunk, summary="Raw Segment Content", pre=True)

def parse_and_generate_html(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    output_path = filepath + ".html"
    
    try:
        with logtree.init_trace(f"Trajectory View: {os.path.basename(filepath)}", path=output_path):
            
            # Split by "---- datum ----"
            # Note: The file might start with content before the first "---- datum ----".
            # We treat that as the first chunk.
            chunks = content.split("---- datum ----")
            
            logtree.log_text(f"Found {len(chunks)} segments (separated by '---- datum ----').")

            for i, chunk in enumerate(chunks):
                process_chunk(chunk, i)
            
        print(f"Generated HTML view: {output_path}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to generate HTML: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View RLM trajectory logs as HTML.")
    parser.add_argument("file", help="Path to the txt/log file")
    args = parser.parse_args()
    
    parse_and_generate_html(args.file)
