---
name: readme-generator
description: This skill should be used when users want to create or improve README.md files for their projects. It generates professional documentation following the Deep Insight/Strands SDK style - comprehensive yet focused, with clear structure and practical examples.
---

# README Generator

## Overview

Generate professional README.md files that follow the Deep Insight/Strands SDK style: clear structure, progressive disclosure, and balanced depth. This skill uses Claude Code's native tools (Read, Glob, Grep) to explore codebases and creates user-friendly documentation.

**Target style**: Deep Insight README - professional, visual, and user-focused.

## When to Use This Skill

Use this skill when:
- Creating a new README.md from scratch
- Improving an existing README with better structure and balance
- Adopting the Deep Insight/Strands SDK documentation style
- Converting technical documentation to user-friendly format

Do NOT use this skill for:
- API-only documentation (use API doc generators instead)
- Internal technical specs (use architectural docs instead)

## The Deep Insight README Pattern

### Structure
1. **Center-aligned header** with logo, title, tagline, badges, and quick navigation links
2. **Latest News** - Recent updates and releases
3. **Why [Project]?** - Value proposition with key benefits
4. **Quick Start** - Get running in 2 minutes (install + basic example)
5. **Demo** - Video/screenshots with sample outputs
6. **Installation** - Detailed setup instructions
7. **Architecture** - System overview with diagrams
8. **Contributing** - Brief welcome with contribution areas
9. **License** - Clear license type
10. **Acknowledgments/Contributors** - Credits and team info

### Key Principles
- **Visual first**: Logo, centered layout, badges, architecture diagrams
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
- Assets: logos, screenshots, diagrams in assets/
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

### Step 2: Build the README

Create sections following the Deep Insight pattern:

#### 1. Header Section with Logo

```markdown
<div align="center">
  <div>
    <img src="./assets/project_logo.png" alt="Project Name" width="110px" height="210px">
  </div>

  <h1 style="margin-top: 10px;">Project Name</h1>

  <h2>Concise value proposition in one sentence</h2>

  <div align="center">
    <a href="https://github.com/user/repo/graphs/commit-activity"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/user/repo"/></a>
    <a href="https://github.com/user/repo/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"/></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.12+-blue.svg"/></a>
  </div>

  <p>
    <a href="#why-project">Why Project?</a>
    ◆ <a href="#quick-start">Quick Start</a>
    ◆ <a href="#demo">Demo</a>
    ◆ <a href="#installation">Installation</a>
    ◆ <a href="#architecture">Architecture</a>
  </p>
</div>
```

**Key points**:
- Logo at the top, adjust size as needed
- Title with reduced margin (`margin-top: 10px`)
- Navigation links with ◆ separator

#### 2. Latest News

Show recent updates in reverse chronological order:

```markdown
## *Latest News* 🔥

- **[2025/10]** Released [Project Workshop](link) (Korean)
- **[2025/10]** Added support for Claude Sonnet 4.5 with enhanced reasoning capabilities
- **[2025/09]** Released Project framework with multi-agent architecture
```

#### 3. Why [Project]?

Value proposition with key benefits:

```markdown
## Why Project Name?

Brief description of the transformation or value provided.

- **🎨 Benefit 1** - Description
- **🔒 Benefit 2** - Description
- **🤖 Benefit 3** - Description
- **📊 Benefit 4** - Description
- **🚀 Benefit 5** - Description
```

#### 4. Quick Start

Minimal commands to get running:

```markdown
## Quick Start

\`\`\`bash
# 1. Clone and setup
git clone repo-url
cd project-dir
cd setup/ && ./create-uv-env.sh env-name 3.12 && cd ..

# 2. Run your analysis
uv run python main.py --user_query "Your task here"
\`\`\`

> **Note**: Requires Python 3.12+ and AWS credentials configured (tested in us-west-2 region).
```

#### 5. Demo

Video and sample outputs:

```markdown
## Demo

### Use Case Title

> **Task**: "Detailed task description"

[▶️ Watch Full Demo on YouTube](video-url)

### Sample Outputs

📄 [Output 1](link) | 📄 [Output 2](link)
```

#### 6. Installation

Complete setup with configuration:

```markdown
## Installation

### Environment Setup

\`\`\`bash
# Navigate to setup directory
cd setup/

# Create UV environment
./create-uv-env.sh project-name 3.12

# Return to project root and run
cd ..
uv run python main.py --user_query "Your request here"
\`\`\`

### Configure AWS Credentials

**Option 1: AWS CLI (Recommended)**

\`\`\`bash
aws configure
\`\`\`

**Option 2: Environment Variables**

\`\`\`bash
export AWS_REGION=us-west-2
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key

# OR use .env file
cp .env.example .env
# Edit .env with your AWS credentials
\`\`\`
```

#### 7. Architecture

System overview with centered diagram:

```markdown
## Architecture

### System Overview

<div align="center">
  <img src="./assets/architecture.png" alt="Project Architecture" width="750">
</div>

### Architecture Details

Include text diagrams or additional details as needed.
```

#### 8. Contributing

```markdown
## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Quick Start for Contributors

\`\`\`bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/repo.git
cd repo-path

# Follow installation steps above

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes, test, then commit and push
git add .
git commit -m "Add feature: description"
git push origin feature/your-feature-name
\`\`\`

### Contribution Areas

- **Area 1**: Description
- **Area 2**: Description
- **Area 3**: Description
```

#### 9. License

```markdown
## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
```

#### 10. Acknowledgments/Contributors

```markdown
## Acknowledgments

### Philosophy

> **"Project Philosophy"**

Brief description of project philosophy or approach.

## Contributors

- **Name** - Title
  - [Email](mailto:email) | [LinkedIn](url) | [GitHub](url)

---

<div align="center">
  <p>
    <strong>Built with ❤️ by Team Name</strong><br>
    <sub>Project mission statement</sub>
  </p>
</div>
```

## Writing Guidelines

### Visual Elements
- **Logo**: Size to 110x210px or adjust proportionally
- **Images**: Center-align with `<div align="center">`, size to ~750px width
- **Badges**: Use relevant badges only (commit activity, license, Python version)
- **Navigation**: Use ◆ separator between links

### Content
- **Latest News**: Most recent first, use `[YYYY/MM]` format
- **Benefits**: Use emojis for visual appeal
- **Code blocks**: Always specify language
- **Links**: Descriptive text, not "click here"

### Structure
- Center-align header section
- Progressive disclosure (quick value → details)
- Clear heading hierarchy (H1 → H2 → H3)
- Keep paragraphs short (3-4 lines max)

## Best Practices

1. **Follow the Deep Insight pattern** - Visual, professional structure
2. **Use center alignment** - Header and diagrams
3. **Include logo** - Brand identity at top
4. **Show real examples** - Actual commands and outputs
5. **Link to resources** - Videos, workshops, sample outputs
6. **Credit contributors** - Team info at bottom
7. **Add Latest News** - Keep users informed of updates

## Common Pitfalls to Avoid

- Missing logo or visual elements
- Not center-aligning header
- Outdated "Latest News" section
- Missing demo video or screenshots
- Generic placeholder text
- Broken internal links
- Inconsistent formatting

## Validation

Before finalizing, verify:

1. **Visual appeal**: Logo, centered header, proper spacing
2. **Completeness**: All essential sections present
3. **Accuracy**: All commands and links work
4. **Clarity**: Non-technical user can follow
5. **Style**: Matches Deep Insight pattern

A well-written README enables users to:
- Understand what it does in 30 seconds
- See visual proof (logo, diagrams, demos)
- Get it running in 2-5 minutes
- Find detailed resources if needed
- Feel confident about the project's quality
