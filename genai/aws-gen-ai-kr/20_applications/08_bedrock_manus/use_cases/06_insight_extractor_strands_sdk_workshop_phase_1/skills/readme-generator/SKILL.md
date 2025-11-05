---
name: readme-generator
description: This skill should be used when users want to create or improve README.md files for their projects. It analyzes codebases automatically and generates comprehensive, user-friendly documentation following industry best practices.
---

# README Generator

## Overview

Generate comprehensive README.md files by automatically analyzing project codebases and following established documentation patterns. This skill uses a hybrid approach: automatically extract project information, then ask users only for missing details.

Target audience: General users who need clear installation and usage instructions (not overly technical).

## When to Use This Skill

Use this skill when:
- Creating a new README.md from scratch
- Improving an existing README with better structure
- Standardizing documentation across multiple projects
- Converting technical documentation to user-friendly format

Do NOT use this skill for:
- API-only documentation (use API doc generators instead)
- Internal technical specs (use architectural docs instead)

## README Generation Process

### Step 1: Analyze Project Structure

Execute the project analysis script to automatically gather information:

```bash
python skills/readme-generator/scripts/analyze_project.py <project_root_path>
```

The script will extract:
- Entry point files (main.py, app.py, etc.)
- Dependencies and frameworks (from pyproject.toml, requirements.txt)
- Directory structure (src/, setup/, etc.)
- Existing documentation files (CLAUDE.md, docs/, etc.)
- Configuration files (.env.example, config files)

Review the JSON output to understand what information is available.

### Step 2: Gather Missing Information

Based on the automatic analysis, identify gaps and ask the user ONLY for:

**Essential information** (if not detectable):
- Project purpose and main value proposition
- Key features or capabilities
- Target user audience

**Optional information** (ask if relevant):
- Demo video or screenshot URLs
- Special installation requirements
- Known issues or limitations
- Contribution guidelines

Use conversational questions, one at a time. Example:
- "What is the main purpose of this project?"
- "What are the 3-5 key features users should know about?"

### Step 3: Generate Core Sections

Create README sections in this recommended order (funnel approach):

1. **Header Section**
   - Project title
   - Tagline (1 sentence description)
   - Badges (optional: build status, license)
   - Language switcher (if multilingual)

2. **Quick Start / Demo**
   - Embed demo video or GIF if available
   - Minimal example showing the project in action
   - Link to dataset/sample files if applicable

3. **Table of Contents** (auto-generate if README is long)

4. **Overview / Introduction**
   - 2-3 paragraphs explaining what the project does
   - Why it exists (problem it solves)
   - Key benefits

5. **Features**
   - Bulleted list of main capabilities
   - Group by category if many features
   - Use clear, user-centric language

6. **Installation**
   - Prerequisites (Python version, system dependencies)
   - Step-by-step installation commands
   - Provide multiple options if available (pip, conda, docker)
   - Include verification steps

7. **Usage**
   - Basic usage example with actual commands
   - Common use cases
   - Configuration options
   - Link to advanced usage or tutorials

8. **Architecture** (optional, if helpful)
   - High-level diagram if available
   - Brief component descriptions
   - Do NOT include implementation details

9. **Troubleshooting** (if common issues exist)
   - FAQ format
   - Known issues with workarounds

10. **Contributing** (if open to contributions)
    - How to set up dev environment
    - How to submit changes
    - Code style guidelines

11. **License**
    - License type (MIT, Apache, etc.)
    - Copyright holder

12. **Acknowledgments / Credits**
    - Dependencies or libraries used
    - Inspiration or related projects
    - Authors and maintainers

Reference the section templates for detailed guidelines: `references/section_templates.md`

### Step 4: Apply User-Friendly Writing Principles

Follow these principles from `references/readme_best_practices.md`:

**Clarity**:
- Use simple, direct language
- Avoid jargon unless necessary (define when used)
- Write in active voice
- Use concrete examples instead of abstract descriptions

**Structure**:
- Maintain clear heading hierarchy (H1 → H2 → H3)
- Use consistent formatting
- Keep paragraphs short (3-4 lines max)
- Use bullet points for lists

**Actionability**:
- Provide copy-paste ready commands
- Include expected outputs
- Use actual file paths and values (not placeholders)
- Specify command execution directory

**Visual Elements**:
- Use code blocks with syntax highlighting
- Add architecture diagrams if helpful
- Include badges for status indicators
- Use emojis sparingly (only for visual categorization)

### Step 5: Format and Validate

**Formatting checklist**:
- [ ] All code blocks have language tags (```python, ```bash)
- [ ] Links are valid and follow format [text](url)
- [ ] Headings follow proper hierarchy (no skipped levels)
- [ ] Lists use consistent bullet style
- [ ] Table of contents matches section headers
- [ ] No broken internal references

**Content checklist**:
- [ ] Installation steps are complete and ordered
- [ ] All commands specify working directory
- [ ] Prerequisites are listed before installation
- [ ] At least one usage example is included
- [ ] License information is present
- [ ] Contact/contribution info is available

**User-friendliness checklist**:
- [ ] Non-technical user can install and run
- [ ] No unexplained technical terms
- [ ] Commands include expected outputs
- [ ] Troubleshooting covers common issues

## Section Guidelines

### Installation Section Best Practices

Focus on getting users up and running quickly:

```markdown
## Installation

### Prerequisites
- Python 3.12 or higher
- UV package manager (recommended) or Conda

### Quick Setup

**Option 1: Using UV (Recommended)**
```bash
# Navigate to setup directory
cd setup/

# Create environment with Python 3.12
./create-uv-env.sh project-name 3.12

# Return to project root and run
cd ..
uv run python main.py
```

**Option 2: Using Conda**
```bash
# Create conda environment
conda create -n project-name python=3.12 -y
conda activate project-name

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Verify Installation
```bash
python --version  # Should show Python 3.12.x
python -c "import package_name; print('Success!')"
```
```

Key elements:
- Clear prerequisites list
- Multiple installation paths
- Actual commands with comments
- Verification steps
- Specify working directories

### Usage Section Best Practices

Show concrete examples:

```markdown
## Usage

### Basic Execution

Run with default settings:
```bash
python main.py
```

Run with custom input:
```bash
python main.py --user_query "Analyze sales data for Q4"
```

### Configuration

Edit `.env` file for custom settings:
```bash
# Copy example configuration
cp .env.example .env

# Edit with your settings
# AWS_REGION=us-west-2
# MODEL_ID=claude-sonnet-4
```

### Output

Results are saved to `artifacts/` directory:
- `analysis_report.pdf` - Main analysis report
- `visualizations/` - Generated charts and graphs
```

Key elements:
- Start with simplest example
- Progress to advanced usage
- Show actual commands and file paths
- Explain where outputs go
- Include configuration examples

## Resources

### Bundled Resources

- **scripts/analyze_project.py** - Automated project structure analyzer
  - Usage: `python scripts/analyze_project.py <project_path>`
  - Output: JSON with project metadata

- **references/section_templates.md** - Templates for all README sections
  - Includes structure and examples for each section
  - Copy-paste ready templates

- **references/readme_best_practices.md** - Industry standards guide
  - Writing principles for clarity
  - Formatting conventions
  - User-friendly documentation tips

### Example Reference

See the current project's README.md as a reference example:
`/home/ubuntu/projects/aws-ai-ml-workshop-kr/genai/aws-gen-ai-kr/20_applications/08_bedrock_manus/use_cases/06_insight_extractor_strands_sdk_workshop_phase_1/README.md`

This README demonstrates:
- Clear funnel structure (overview → quick start → details)
- Multiple installation options
- Visual elements (architecture diagram)
- User-friendly language
- Comprehensive but not overwhelming

## Best Practices Summary

1. **Start with automation** - Use the analysis script to gather facts
2. **Ask minimal questions** - Only request information that cannot be detected
3. **Follow the funnel** - Most important info first, details later
4. **Write for users, not developers** - Assume no prior technical knowledge
5. **Be specific** - Use actual commands, not generic placeholders
6. **Stay concise** - Every section should have a clear purpose
7. **Test instructions** - Ensure installation/usage steps actually work
8. **Update regularly** - README should match current codebase state

## Common Pitfalls to Avoid

- **Over-explaining** - Don't document every implementation detail
- **Vague instructions** - "Install dependencies" is not enough; show HOW
- **Broken examples** - Test all commands before including them
- **Missing prerequisites** - List ALL required software/versions
- **Assuming knowledge** - Define technical terms or link to explanations
- **UI-heavy content** - Avoid extensive UI screenshots (use sparingly)
- **Outdated info** - Remove obsolete sections, update version numbers

## Validation

Before finalizing the README, verify:

1. **Completeness**: All essential sections present
2. **Accuracy**: All commands and paths are correct
3. **Clarity**: Non-technical user can follow instructions
4. **Consistency**: Formatting and style are uniform
5. **Accessibility**: Links work, images load, code renders properly

A well-written README should enable a new user to:
- Understand what the project does in 30 seconds
- Install and run it in 5 minutes
- Find advanced documentation if needed
