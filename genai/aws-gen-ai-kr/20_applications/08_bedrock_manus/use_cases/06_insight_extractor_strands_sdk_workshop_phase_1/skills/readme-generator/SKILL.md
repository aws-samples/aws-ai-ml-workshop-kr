---
name: readme-generator
description: This skill should be used when users want to create or improve README.md files for their projects. It generates professional, balanced documentation following the Strands SDK style - comprehensive yet focused, with clear structure and practical examples.
---

# README Generator

## Overview

Generate professional README.md files that follow the Strands SDK style: clear structure, progressive disclosure, and balanced depth. This skill uses Claude Code's native tools (Read, Glob, Grep) to explore codebases and creates user-friendly documentation through guided conversation.

**Target style**: Strands SDK README - not too minimal, not too verbose, just right.

## When to Use This Skill

Use this skill when:
- Creating a new README.md from scratch
- Improving an existing README with better structure and balance
- Adopting the Strands SDK documentation style
- Converting technical documentation to user-friendly format

Do NOT use this skill for:
- API-only documentation (use API doc generators instead)
- Internal technical specs (use architectural docs instead)

## The Strands SDK README Pattern

The gold standard we follow has these characteristics:

### Structure
1. **Center-aligned header** with logo, badges, and quick navigation links
2. **Feature Overview** - Brief, scannable (3-5 bullet points)
3. **Quick Start** - Get running in 2 minutes (install + basic example)
4. **Installation** - Detailed setup instructions
5. **Features at a Glance** - Code examples for each major feature (optional for complex projects)
6. **Documentation** - Links to comprehensive docs
7. **Contributing** - Brief welcome with link to details
8. **License** - Clear license type
9. **Security** - Security policy if applicable

### Key Principles
- **Balanced depth**: Substantial enough to be useful, focused enough to stay readable
- **Progressive disclosure**: Quick value at top, details further down
- **Code-first**: Show working examples, not just descriptions
- **Professional yet accessible**: Clear language without excessive jargon

## README Generation Workflow

### Step 1: Explore the Codebase

Use Claude Code's native tools to gather essential information:

**Project structure:**
```
Use Glob to find key files:
- Entry points: main.py, app.py, *.ipynb
- Config: pyproject.toml, requirements.txt, .env.example
- Docs: CLAUDE.md, existing README, CONTRIBUTING.md
```

**Dependencies and frameworks:**
```
Use Read to examine:
- pyproject.toml or requirements.txt for dependencies
- Key imports in main files to identify frameworks
- .env.example for required configuration
```

**Key features:**
```
Use Read to understand:
- Main entry point logic
- Command-line arguments or API endpoints
- Output artifacts or deliverables
```

**Important**: Be selective. Only gather information that will appear in the README. Don't extract implementation details.

### Step 2: Gather User Context

Ask the user conversational questions (one at a time) for information you can't detect:

**Essential questions** (if not obvious from code):
- "What's the main purpose of this project?" (for tagline)
- "What problem does it solve?" (for Overview section)
- "Who is the target audience?" (general users, developers, data scientists)

**Optional questions** (if relevant):
- "Do you have a demo video or screenshot URL?"
- "Are there any special prerequisites or system dependencies?"
- "What's the GitHub repository URL?" (for badges)

**Example approach**:
```
"I can see this is a multi-agent system using Strands SDK for data analysis.
What would you say is the main value proposition for users?
For example: 'Automate data analysis with AI agents' or 'Generate reports from natural language queries'"
```

### Step 3: Build the README

Create sections following the Strands SDK pattern:

#### 1. Header Section

Use center-aligned format with badges and navigation:

```markdown
<div align="center">
  <h1>Project Name</h1>

  <h2>Concise value proposition in one sentence</h2>

  <div align="center">
    <a href="https://github.com/user/repo/graphs/commit-activity"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/user/repo"/></a>
    <a href="https://github.com/user/repo/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/user/repo"/></a>
    <a href="https://github.com/user/repo/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/user/repo"/></a>
    <a href="https://python.org"><img alt="Python" src="https://img.shields.io/badge/python-3.12+-blue.svg"/></a>
  </div>

  <p>
    <a href="#installation">Installation</a>
    ◆ <a href="#usage">Usage</a>
    ◆ <a href="#features-at-a-glance">Features</a>
    ◆ <a href="#documentation">Documentation</a>
  </p>
</div>
```

**Key points**:
- Title (H1) and value proposition (H2)
- Relevant badges only (commit activity, issues, license, Python version)
- Quick navigation links with ◆ separator
- Links should point to sections that exist in your README

#### 2. Feature Overview

Brief introduction with 3-5 key capabilities:

```markdown
## Feature Overview

Brief 1-2 sentence description of what the project does.

- **Capability 1**: Brief description focusing on user benefit
- **Capability 2**: Brief description focusing on user benefit
- **Capability 3**: Brief description focusing on user benefit
- **Capability 4**: Brief description focusing on user benefit
```

**Guidelines**:
- Focus on WHAT it does, not HOW
- User benefits, not technical implementation
- Each bullet under 15 words

#### 3. Quick Start

Minimal commands to get running fast:

```markdown
## Quick Start

\`\`\`bash
# Install
pip install package-name
# OR for development
git clone repo-url
cd project-name
./setup.sh
\`\`\`

\`\`\`python
# Basic usage
from package import Module
result = Module().run("your query")
\`\`\`

> **Note**: Requires Python 3.12+ and AWS credentials configured.
```

**Guidelines**:
- Keep it under 10 lines of code
- Show the absolute minimum to see value
- Include prerequisite note inline

#### 4. Installation (Detailed)

Complete setup instructions with multiple options if applicable:

```markdown
## Installation

Ensure you have Python 3.10+ installed, then:

\`\`\`bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
\`\`\`

### Configuration

Configure environment variables:

\`\`\`bash
# Copy template
cp .env.example .env

# Edit with your settings
# AWS_REGION=us-west-2
# MODEL_ID=claude-sonnet-4
\`\`\`

### Verify Installation

\`\`\`bash
python --version  # Should show Python 3.10+
python -c "import package_name; print('Success!')"
\`\`\`
```

**Guidelines**:
- Show complete steps from clone to running
- Provide multiple paths (UV, pip, conda) if applicable
- Include verification steps
- Specify working directory when needed

#### 5. Features at a Glance (Optional)

For complex projects with multiple capabilities, show code examples for each:

```markdown
## Features at a Glance

### Python-Based Tools

Easily build tools using Python decorators:

\`\`\`python
from project import tool

@tool
def analyze(text: str) -> dict:
    """Analyze text and return insights."""
    return {"word_count": len(text.split())}
\`\`\`

### Multi-Model Support

Support for various model providers:

\`\`\`python
from project import Agent, BedrockModel

model = BedrockModel(model_id="claude-sonnet-4")
agent = Agent(model=model)
response = agent("Analyze this data")
\`\`\`

### Streaming Output

Real-time progress updates:

\`\`\`python
for event in agent.stream("Long running task"):
    print(event)
\`\`\`
```

**Guidelines**:
- Each feature gets H3 heading + brief intro + code example
- Keep examples practical and runnable
- Link to detailed docs for more info

#### 6. Documentation (for complex projects)

Link to comprehensive documentation:

```markdown
## Documentation

For detailed guidance, explore our documentation:

- [User Guide](url) - Getting started and core concepts
- [API Reference](url) - Complete API documentation
- [Examples](url) - Sample projects and use cases
- [Deployment Guide](url) - Production deployment
```

**Guidelines**:
- Organize by audience/purpose
- Brief description for each link
- Only include if you have extensive external docs

#### 7. Contributing

Brief welcome statement:

```markdown
## Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details on:
- Reporting bugs & features
- Development setup
- Submitting Pull Requests
- Code of Conduct
```

**Guidelines**:
- Keep it brief in README
- Link to CONTRIBUTING.md for details
- Mention key contribution areas

#### 8. License

Clear license statement:

```markdown
## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
```

#### 9. Security (if applicable)

```markdown
## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for information on reporting security issues.
```

### Step 4: Polish and Refine

Apply these final touches:

**Formatting checklist**:
- [ ] All code blocks have language tags (\`\`\`python, \`\`\`bash)
- [ ] Links are valid and use descriptive text
- [ ] Headings follow proper hierarchy (H1 → H2 → H3)
- [ ] Internal anchor links match section headings (lowercase with hyphens)
- [ ] No broken references

**Content checklist**:
- [ ] Value proposition is clear and compelling
- [ ] At least one working code example in Quick Start
- [ ] Installation steps are complete and ordered
- [ ] No unexplained technical jargon
- [ ] Contact/support info is available

**Balance checklist**:
- [ ] Not too minimal (has substance and examples)
- [ ] Not too verbose (stays focused on essentials)
- [ ] Follows progressive disclosure (quick value → details)
- [ ] Matches Strands SDK style

## Writing Guidelines

### Clarity
- Use simple, direct language
- Define technical terms when first used
- Write in active voice ("The agent processes data" not "Data is processed")
- Use concrete examples over abstract descriptions

### Structure
- Maintain clear heading hierarchy
- Use consistent formatting throughout
- Keep paragraphs short (3-4 lines max)
- Use bullet points for lists

### Code Examples
- Always specify language for syntax highlighting
- Include comments for complex commands
- Use real values, not placeholders (with notes on what to change)
- Test all commands before including

### Visual Elements
- Use center alignment for header section
- Include badges for project status
- Add navigation links with ◆ separator
- Consider adding architecture diagram if helpful

## Best Practices

### Structure and Organization
1. **Follow the Strands SDK pattern** - Proven, professional structure
2. **Progressive disclosure** - Most important info first
3. **Keep it balanced** - Comprehensive yet focused
4. **Use code examples liberally** - Show, don't just tell

### Content and Writing
5. **Start with codebase exploration** - Use Read/Glob/Grep to gather facts
6. **Ask minimal questions** - Only request what you can't detect
7. **Write for users** - Clear, accessible language
8. **Be specific** - Actual commands, not placeholders

### Visual and Formatting
9. **Center-align header** - Professional, polished appearance
10. **Include status badges** - Shows project health
11. **Add navigation links** - Easy access to key sections
12. **Proper markdown** - Syntax highlighting, alt text, proper hierarchy

## Common Pitfalls to Avoid

### Content Issues
- **Too minimal** - Just a title and install command isn't enough
- **Too verbose** - Don't document every detail in README
- **Vague instructions** - Show specific commands, not "install dependencies"
- **Assuming knowledge** - Define terms, list prerequisites

### Structure Issues
- **Poor hierarchy** - Most important info should come first
- **Missing Quick Start** - Users need working code fast
- **No examples** - Show working code, not just API docs
- **Broken examples** - Test all commands

### Visual Issues
- **Wall of text** - Use headings, bullets, code blocks
- **No status indicators** - Add badges for project health
- **Missing navigation** - Add quick links at top

## Examples and References

**Gold Standard**: Strands Agents SDK README
- Perfect balance of depth and focus
- Clear progressive disclosure
- Excellent code examples
- Professional presentation

**Key characteristics to emulate**:
- Center-aligned header with badges
- Feature Overview before diving into details
- Quick Start gets you running in 2 minutes
- Features at a Glance shows practical usage
- Documentation links for deeper exploration
- Clean, professional appearance

Use this as your template when generating READMEs.

## Validation

Before finalizing, verify:

1. **Completeness**: All essential sections present
2. **Accuracy**: All commands and paths work
3. **Clarity**: Non-technical user can follow instructions
4. **Balance**: Not too minimal, not too verbose
5. **Style**: Matches Strands SDK pattern

A well-written README enables users to:
- Understand what it does in 30 seconds
- Get it running in 2-5 minutes
- Find detailed docs if needed
- Feel confident about the project's quality
