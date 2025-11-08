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

Minimal commands to get running (2-3 steps max):

```markdown
## Quick Start

\`\`\`bash
# 1. Clone and setup environment
git clone https://github.com/user/repo.git
cd repo-directory
./setup.sh  # or your setup command

# 2. Configure credentials/environment (if needed)
cp .env.example .env
# Edit .env with your configuration

# 3. Run basic example
python main.py --example "basic task"
\`\`\`

> **Prerequisites**: List key requirements (e.g., Python 3.12+, API keys, system dependencies)
>
> **Need more options?** See [Installation](#installation) section below for detailed setup instructions and alternative configuration methods.
```

#### 5. Demo

Video and sample outputs:

```markdown
## Demo

### Use Case Title

> **Task**: "Detailed task description"
>
> **Workflow**: Input (data source) → Process (natural language prompt) → Output (deliverables with analysis)

[▶️ Watch Full Demo on YouTube](video-url)

### Sample Outputs

📄 [Output 1](link) | 📄 [Output 2](link)
```

#### 6. Installation

Complete setup with configuration:

```markdown
## Installation

This section provides detailed installation instructions and alternative configuration options. For a quick setup, see [Quick Start](#quick-start) above.

### Environment Setup

\`\`\`bash
# Clone the repository
git clone https://github.com/user/repo.git
cd repo-directory

# Install dependencies (choose your method)
pip install -r requirements.txt
# OR
poetry install
# OR
./setup.sh
\`\`\`

The setup automatically:
- Installs required dependencies
- Sets up virtual environment
- Configures initial settings

### Configuration

Provide multiple configuration options for flexibility:

**Option 1: Configuration File (Recommended)**

\`\`\`bash
cp config.example.yaml config.yaml
# Edit config.yaml with your settings
\`\`\`

**Option 2: Environment Variables**

\`\`\`bash
# Direct export (session-based)
export API_KEY=your_api_key
export ENVIRONMENT=production
\`\`\`

**Option 3: .env File (Persistent)**

\`\`\`bash
# Copy example file and edit
cp .env.example .env
# Edit .env with your configuration
\`\`\`

> **Security Note**: Never commit files with real credentials to version control. Sensitive files should be in \`.gitignore\`.
```

#### 7. Architecture

System overview with visual diagram and optional text-based architecture:

```markdown
## Architecture

### System Overview

<div align="center">
  <img src="./assets/architecture.png" alt="Project Architecture" width="750">
</div>

### Component Architecture (Optional)

For complex systems, include text-based diagrams to explain flow:

\`\`\`
┌─────────────────────────────────────────────────────────┐
│                     User Input                          │
│                   (Entry Point)                         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  COMPONENT A (Primary Handler)                          │
│  • Responsibility 1                                     │
│  • Responsibility 2                                     │
│  • Responsibility 3                                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  COMPONENT B (Processor)                                │
│  • Processing step 1                                    │
│  • Processing step 2                                    │
└──────────┬──────────┬──────────┬────────────────────────┘
           │          │          │
     ┌─────┘    ┌─────┘    ┌─────┘
     ▼          ▼          ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ MODULE1 │ │ MODULE2 │ │ MODULE3 │
│         │ │         │ │         │
│ Task A  │ │ Task B  │ │ Task C  │
└─────────┘ └─────────┘ └─────────┘
\`\`\`

### Key Design Decisions

Explain architectural choices:
- **Pattern Used**: Description of architectural pattern (e.g., microservices, event-driven)
- **Technology Stack**: Key frameworks and libraries
- **Scalability**: How the system scales
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

- **Feature Development**: Add new features and capabilities
- **Bug Fixes**: Fix issues and improve stability
- **Documentation**: Improve guides, examples, and tutorials
- **Testing**: Add tests and improve test coverage
- **Performance**: Optimize code and improve efficiency
- **Design**: Improve UI/UX and visual elements
```

#### 9. License

```markdown
## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
```

#### 10. Acknowledgments/Contributors

```markdown
## Acknowledgments

### Philosophy (Optional)

> **"Your project philosophy or motto"**

Brief description of project values, inspiration, or approach.

## Contributors

**Option 1: Simple List**
- **Name** - Role
  - [Email](mailto:email) · [GitHub](https://github.com/username)

**Option 2: Table Format (for multiple contributors)**

| Name | Role | Contact |
|------|------|---------|
| **Person 1** | Lead Developer | [Email](mailto:email1) · [LinkedIn](url) · [GitHub](url) |
| **Person 2** | Contributor | [Email](mailto:email2) · [LinkedIn](url) · [GitHub](url) |
| **Person 3** | Documentation | [Email](mailto:email3) · [LinkedIn](url) · [GitHub](url) |

---

<div align="center">
  <p>
    <strong>Built with ❤️ by [Team Name]</strong><br>
    <sub>Your project mission or tagline</sub>
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
