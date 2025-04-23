import json
import os
import re
from datetime import datetime

tool_list = [
    {
        "toolSpec": {
            "name": "create_todo",
            "description": "Creates a new todo.md file based on the provided plan information. Use this tool to initialize a structured task list at the beginning of a project.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "plan_json": {
                            "type": "string",
                            "description": "JSON string representing the plan with steps and task information"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Path where the todo.md file should be created (default: todo.md)"
                        }
                    },
                    "required": ["plan_json"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "update_task_status",
            "description": "Updates the status of a specific task in the todo.md file. Use this to mark tasks as completed or revert them to pending status.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "The description of the task to update (must match exactly or provide a unique part)"
                        },
                        "completed": {
                            "type": "boolean",
                            "description": "Whether to mark the task as completed (true) or pending (false)"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Path to the todo.md file (default: todo.md)"
                        }
                    },
                    "required": ["task_description", "completed"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "rebuild_todo",
            "description": "Rebuilds the todo.md file based on an updated plan while preserving the completion status of existing tasks. Use this when the project plan has changed significantly.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "updated_plan_json": {
                            "type": "string",
                            "description": "JSON string representing the updated plan"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Path to the todo.md file (default: todo.md)"
                        }
                    },
                    "required": ["updated_plan_json"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "calculate_progress",
            "description": "Calculates and reports the current progress based on the todo.md file. Use this to get a summary of task completion status.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the todo.md file (default: todo.md)"
                        },
                        "include_details": {
                            "type": "boolean",
                            "description": "Whether to include detailed information about each task (default: false)"
                        }
                    },
                    "required": []
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "add_task",
            "description": "Adds a new task to the existing todo.md file. Use this when you need to add additional tasks that weren't in the original plan.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "Description of the new task to add"
                        },
                        "agent_name": {
                            "type": "string",
                            "description": "Name of the agent responsible for this task"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Path to the todo.md file (default: todo.md)"
                        },
                        "section": {
                            "type": "string",
                            "description": "Section where the task should be added (optional)"
                        }
                    },
                    "required": ["task_description", "agent_name"]
                }
            }
        }
    }
]

task_tracker_tool_config = {
    "tools": tool_list,
    # "toolChoice": {
    #    "tool": {
    #        "name": "summarize_email"
    #    }
    # }
}

def process_task_tracker_tool(tool) -> dict:
    """Process a task tracker tool invocation
    
    Args:
        tool: Tool definition including name and input
        
    Returns:
        Result of the tool invocation as a dictionary
    """
    
    tool_name, tool_input = tool['name'], tool['input']
    
    if tool_name == "create_todo":
        results = handle_create_todo(tool_input)
    elif tool_name == "update_task_status":
        results = handle_update_task_status(tool_input)
    elif tool_name == "rebuild_todo":
        results = handle_rebuild_todo(tool_input)
    elif tool_name == "calculate_progress":
        results = handle_calculate_progress(tool_input)
    elif tool_name == "add_task":
        results = handle_add_task(tool_input)
    else:
        results = f"Unknown task tracker tool: {tool_name}"
        
    tool_result = {
        "toolUseId": tool.get('toolUseId', 'unknown'),
        "content": [{"json": {"text": results}}]
    }
    
    return {"role": "user", "content": [{"toolResult": tool_result}]}

def handle_create_todo(tool_input) -> str:
    """Creates a new todo.md file based on the provided plan
    
    Args:
        tool_input: Dictionary containing 'plan_json' and optional 'file_path'
        
    Returns:
        Result message as a string
    """
    # Get parameters
    plan_json = tool_input.get("plan_json", "{}")
    file_path = tool_input.get("file_path", "todo.md")
    
    try:
        # Parse the plan JSON
        plan = json.loads(plan_json)
        
        # Create todo.md content
        content = f"# {plan.get('title', 'Task List')}\n\n"
        content += f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add tasks from the plan
        for i, step in enumerate(plan.get('steps', [])):
            agent = step.get('agent_name', '')
            title = step.get('title', '')
            description = step.get('description', '')
            
            content += f"## Step {i+1}: {title}\n"
            content += f"- [ ] **Agent:** {agent}\n"
            content += f"- [ ] **Task:** {description}\n\n"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        # Write to file
        with open(file_path, 'w') as f:
            f.write(content)
        
        return f"✅ Todo file successfully created at '{file_path}' with {len(plan.get('steps', []))} tasks."
    
    except json.JSONDecodeError:
        return f"❌ Error: Invalid JSON in plan_json parameter."
    except Exception as e:
        return f"❌ Error creating todo file: {str(e)}"

def handle_update_task_status(tool_input) -> str:
    """Updates the status of a specific task in the todo.md file
    
    Args:
        tool_input: Dictionary containing 'task_description', 'completed', and optional 'file_path'
        
    Returns:
        Result message as a string
    """
    # Get parameters
    task_description = tool_input.get("task_description", "")
    completed = tool_input.get("completed", False)
    file_path = tool_input.get("file_path", "todo.md")
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return f"❌ Error: Todo file '{file_path}' not found."
        
        # Read the current content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find the task in the content
        task_pattern = re.compile(r'- \[([ x])\] \*\*Task:\*\* (.*?)$', re.MULTILINE)
        matches = list(task_pattern.finditer(content))
        
        task_found = False
        updated_content = content
        
        for match in matches:
            current_status = match.group(1)
            current_task = match.group(2)
            
            if task_description in current_task:
                task_found = True
                
                # Current status matches requested status, no change needed
                if (current_status == 'x' and completed) or (current_status == ' ' and not completed):
                    return f"ℹ️ Task '{task_description}' already has the requested status."
                
                # Update the status
                new_status = 'x' if completed else ' '
                timestamp = f" (Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})" if completed else ""
                
                # Remove existing timestamp if changing to not completed
                if not completed and "Completed:" in current_task:
                    current_task = re.sub(r' \(Completed: .*?\)', '', current_task)
                
                old_str = f"- [{current_status}] **Task:** {current_task}"
                new_str = f"- [{new_status}] **Task:** {current_task}{timestamp}"
                updated_content = updated_content.replace(old_str, new_str)
                
                break
        
        if not task_found:
            return f"❌ Error: Task containing '{task_description}' not found in todo file."
        
        # Write the updated content
        with open(file_path, 'w') as f:
            f.write(updated_content)
        
        status_text = "completed" if completed else "pending"
        return f"✅ Task '{task_description}' marked as {status_text}."
    
    except Exception as e:
        return f"❌ Error updating task status: {str(e)}"

def handle_rebuild_todo(tool_input) -> str:
    """Rebuilds the todo.md file based on an updated plan while preserving completion status
    
    Args:
        tool_input: Dictionary containing 'updated_plan_json' and optional 'file_path'
        
    Returns:
        Result message as a string
    """
    # Get parameters
    updated_plan_json = tool_input.get("updated_plan_json", "{}")
    file_path = tool_input.get("file_path", "todo.md")
    
    try:
        # Parse the updated plan JSON
        updated_plan = json.loads(updated_plan_json)
        
        # Check if original file exists and read it
        existing_tasks = {}
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract existing tasks and their completion status
            task_pattern = re.compile(r'- \[([ x])\] \*\*Task:\*\* (.*?)$', re.MULTILINE)
            matches = list(task_pattern.finditer(content))
            
            for match in matches:
                status = match.group(1)
                task = match.group(2)
                # Remove timestamp if present
                task_key = re.sub(r' \(Completed: .*?\)', '', task)
                existing_tasks[task_key] = {
                    'status': status,
                    'full_text': task
                }
        
        # Create new todo.md content
        new_content = f"# {updated_plan.get('title', 'Task List')}\n\n"
        new_content += f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        preserved_count = 0
        new_count = 0
        
        # Add tasks from the updated plan
        for i, step in enumerate(updated_plan.get('steps', [])):
            agent = step.get('agent_name', '')
            title = step.get('title', '')
            description = step.get('description', '')
            
            new_content += f"## Step {i+1}: {title}\n"
            new_content += f"- [ ] **Agent:** {agent}\n"
            
            # Check if this task existed before and preserve its status
            if description in existing_tasks:
                task_info = existing_tasks[description]
                status = task_info['status']
                
                if status == 'x':
                    preserved_count += 1
                    new_content += f"- [x] **Task:** {task_info['full_text']}\n\n"
                else:
                    new_content += f"- [ ] **Task:** {description}\n\n"
            else:
                new_count += 1
                new_content += f"- [ ] **Task:** {description}\n\n"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        # Write to file
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        removed_count = len(existing_tasks) - preserved_count
        
        return (f"✅ Todo file successfully rebuilt at '{file_path}'.\n"
                f"📊 Statistics: {len(updated_plan.get('steps', []))} total tasks, "
                f"{preserved_count} preserved, {new_count} new, {removed_count} removed.")
    
    except json.JSONDecodeError:
        return f"❌ Error: Invalid JSON in updated_plan_json parameter."
    except Exception as e:
        return f"❌ Error rebuilding todo file: {str(e)}"

def handle_calculate_progress(tool_input) -> str:
    """Calculates and reports the current progress based on the todo.md file
    
    Args:
        tool_input: Dictionary containing optional 'file_path' and 'include_details'
        
    Returns:
        Result message as a string with progress information
    """
    # Get parameters
    file_path = tool_input.get("file_path", "todo.md")
    include_details = tool_input.get("include_details", False)
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return f"❌ Error: Todo file '{file_path}' not found."
        
        # Read the content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract tasks and their completion status
        task_pattern = re.compile(r'- \[([ x])\] \*\*Task:\*\* (.*?)$', re.MULTILINE)
        matches = list(task_pattern.finditer(content))
        
        total_tasks = len(matches)
        completed_tasks = sum(1 for match in matches if match.group(1) == 'x')
        
        if total_tasks == 0:
            return "⚠️ No tasks found in the todo file."
        
        progress_percentage = (completed_tasks / total_tasks) * 100
        
        result = f"📊 Progress Report:\n\n"
        result += f"Total Tasks: {total_tasks}\n"
        result += f"Completed: {completed_tasks}\n"
        result += f"Progress: {progress_percentage:.1f}%\n"
        
        # Create a visual progress bar
        progress_bar_length = 20
        filled_length = int(progress_bar_length * completed_tasks / total_tasks)
        progress_bar = '█' * filled_length + '░' * (progress_bar_length - filled_length)
        result += f"[{progress_bar}] {progress_percentage:.1f}%\n\n"
        
        if include_details:
            result += "Task Details:\n\n"
            
            # Find all step titles
            step_pattern = re.compile(r'## Step (\d+): (.*?)$', re.MULTILINE)
            step_matches = list(step_pattern.finditer(content))
            
            current_step = 0
            for match in task_pattern.finditer(content):
                status = match.group(1)
                task = match.group(2)
                
                # Find which step this task belongs to
                for i, step_match in enumerate(step_matches):
                    step_pos = step_match.start()
                    if step_pos < match.start() and (i == len(step_matches) - 1 or step_matches[i + 1].start() > match.start()):
                        current_step = i
                        break
                
                step_num = step_matches[current_step].group(1) if current_step < len(step_matches) else "?"
                step_title = step_matches[current_step].group(2) if current_step < len(step_matches) else "Unknown"
                
                status_symbol = "✅" if status == 'x' else "⬜"
                result += f"{status_symbol} Step {step_num}: {step_title} - {task}\n"
        
        return result
    
    except Exception as e:
        return f"❌ Error calculating progress: {str(e)}"

def handle_add_task(tool_input) -> str:
    """Adds a new task to the existing todo.md file
    
    Args:
        tool_input: Dictionary containing 'task_description', 'agent_name', optional 'file_path', and optional 'section'
        
    Returns:
        Result message as a string
    """
    # Get parameters
    task_description = tool_input.get("task_description", "")
    agent_name = tool_input.get("agent_name", "")
    file_path = tool_input.get("file_path", "todo.md")
    section = tool_input.get("section", "")
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return f"❌ Error: Todo file '{file_path}' not found. Create it first using create_todo."
        
        # Read the current content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Determine where to add the new task
        if section:
            # Try to find the specified section
            section_pattern = re.compile(f"## {re.escape(section)}.*?$", re.MULTILINE)
            section_match = section_pattern.search(content)
            
            if not section_match:
                # If section not found, create it
                next_step_num = len(re.findall(r'## Step \d+:', content)) + 1
                new_section = f"\n## Step {next_step_num}: {section}\n"
                content += new_section
                insert_position = len(content)
            else:
                # Find position after the section heading
                insert_position = section_match.end()
        else:
            # Add to the end of the file
            insert_position = len(content)
        
        # Create the new task entry
        new_task = f"- [ ] **Agent:** {agent_name}\n- [ ] **Task:** {task_description}\n\n"
        
        # Insert the new task at the determined position
        updated_content = content[:insert_position] + "\n" + new_task + content[insert_position:]
        
        # Write the updated content
        with open(file_path, 'w') as f:
            f.write(updated_content)
        
        if section:
            return f"✅ New task '{task_description}' added to section '{section}'."
        else:
            return f"✅ New task '{task_description}' added to the end of todo file."
    
    except Exception as e:
        return f"❌ Error adding task: {str(e)}"