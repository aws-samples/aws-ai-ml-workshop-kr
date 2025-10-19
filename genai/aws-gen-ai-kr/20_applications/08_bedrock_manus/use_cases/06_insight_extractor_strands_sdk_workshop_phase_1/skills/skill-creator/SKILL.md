---
name: skill-creator
description: Comprehensive guide for creating effective Claude Code skills for any domain or task
---

# Skill Creator

This skill provides a complete methodology for creating effective Claude Code skills. Use this guide whenever you need to define a new skill for any task or domain.

## What are Skills in Claude Code?

**Skills** are specialized knowledge modules that extend Claude Code's capabilities for specific tasks or domains. They are invoked using the `Skill` tool and provide:

- **Domain expertise**: Deep knowledge in specific areas (e.g., API documentation, security review, data analysis)
- **Consistent patterns**: Standardized approaches to recurring tasks
- **Reusable guidelines**: Best practices that can be applied across projects
- **Contextual behavior**: Task-specific instructions that override general behavior

### Skills vs. System Prompts

| Aspect | Skills | System Prompts |
|--------|--------|----------------|
| **Purpose** | Task-specific expertise | Agent identity and behavior |
| **Scope** | Narrow, focused domain | Broad, general instructions |
| **Invocation** | On-demand via Skill tool | Always active |
| **Audience** | Claude Code for specific tasks | Claude as an agent |
| **Examples** | "API documentation writer", "Security reviewer" | "You are a coordinator agent" |

**Key Difference**: Skills teach **how to do a specific type of work**. System prompts define **who the agent is and how it behaves generally**.

## Skill File Structure

Every skill is defined in a `SKILL.md` file with this structure:

```markdown
---
name: skill-name
description: Brief description of what this skill does (1-2 sentences)
---

# Skill Title

[Main content organized in sections]
```

### Required Components

1. **YAML Frontmatter**
   - `name`: Kebab-case identifier (e.g., `api-doc-writer`)
   - `description`: Concise summary for Claude Code to understand when to use this skill

2. **Title and Introduction**
   - Clear, descriptive title
   - Brief explanation of the skill's purpose and scope

3. **Core Content**
   - Guidelines, patterns, and instructions
   - Examples demonstrating best practices
   - Templates for common outputs
   - Validation criteria

### Optional but Recommended Components

- **Background/Context**: Domain knowledge needed to understand the skill
- **Step-by-step processes**: Workflows for complex tasks
- **Anti-patterns**: Common mistakes to avoid
- **References**: Links to authoritative sources
- **Checklists**: Validation criteria for task completion

## Principles for Effective Skills

### 1. Clear Scope Definition

**Good - Focused:**
```yaml
name: rest-api-doc-writer
description: Expert guidance for writing RESTful API documentation following OpenAPI 3.0 standards
```

**Bad - Too Broad:**
```yaml
name: documentation-writer
description: Help with writing documentation
```

**Why**: Narrow, well-defined scope makes the skill more useful and easier to invoke correctly.

### 2. Actionable Instructions

**Good - Specific:**
```markdown
## Endpoint Documentation Structure

For each endpoint, include:
1. HTTP method and path
2. Short description (1 sentence)
3. Parameters table with: name, type, required/optional, description
4. Example request with actual JSON
5. Example response with status code and JSON
6. Error responses (4xx, 5xx)
```

**Bad - Vague:**
```markdown
## Writing Endpoints

Document your endpoints clearly with all necessary information.
```

**Why**: Specific, numbered steps are easier to follow and validate.

### 3. Concrete Examples

**Good - Shows, Not Tells:**
```markdown
## Example: POST Endpoint Documentation

### Create User
**POST** `/api/v1/users`

Creates a new user account.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| email | string | Yes | User's email address |
| name | string | Yes | Full name |
| role | string | No | User role (default: 'user') |

**Example Request:**
\`\`\`json
{
  "email": "alice@example.com",
  "name": "Alice Smith",
  "role": "admin"
}
\`\`\`

**Success Response (201 Created):**
\`\`\`json
{
  "id": "usr_123abc",
  "email": "alice@example.com",
  "name": "Alice Smith",
  "role": "admin",
  "created_at": "2025-01-15T10:30:00Z"
}
\`\`\`
```

**Why**: Examples clarify ambiguous instructions and provide concrete patterns to follow.

### 4. Domain-Specific Knowledge

Include specialized knowledge that isn't common:

```markdown
## API Versioning Best Practices

**URL Versioning (Recommended):**
- Use `/api/v1/`, `/api/v2/` in the path
- Clear, visible versioning
- Easy to route and maintain

**Header Versioning:**
- Use `Accept: application/vnd.api+json; version=1`
- Cleaner URLs but harder to test
- Use when URL versioning isn't feasible

**Avoid:**
- Query parameter versioning (`?version=1`) - difficult to cache
- No versioning - breaking changes hurt users
```

### 5. Validation Criteria

End with clear success criteria:

```markdown
## API Documentation Checklist

Before finalizing, verify:
- [ ] All endpoints have HTTP method, path, and description
- [ ] Parameters include type and required/optional status
- [ ] At least one example request and response per endpoint
- [ ] Error responses documented (400, 401, 404, 500)
- [ ] Authentication requirements specified
- [ ] Rate limiting noted if applicable
- [ ] Version number included in all paths
```

## Skill Creation Process

Follow these steps to create a new skill:

### Step 1: Define Purpose and Scope

**Questions to Answer:**
- What specific task or domain does this skill cover?
- What are the boundaries? (What's in scope, what's out of scope?)
- Who would use this skill and when?
- What expertise does it provide that general knowledge doesn't?

**Example:**
```
Purpose: Write security-focused code reviews for Python web applications
Scope: Security vulnerabilities (SQL injection, XSS, auth issues)
Out of Scope: Code style, performance optimization, general bugs
Users: Security engineers, senior developers doing security review
Expertise: OWASP Top 10, Python-specific security patterns, CVE references
```

### Step 2: Gather Core Concepts

**Collect:**
- Key principles and best practices
- Common patterns in the domain
- Terminology and definitions
- Standards or frameworks (e.g., OWASP, REST, OpenAPI)
- Decision criteria and trade-offs

**Organize by:**
- Importance (critical vs. nice-to-have)
- Complexity (foundational vs. advanced)
- Usage frequency (common vs. rare scenarios)

### Step 3: Structure the Skill

**Recommended Sections:**

1. **Introduction** - What this skill does and why it matters
2. **Core Principles** - Fundamental concepts and philosophy
3. **Guidelines** - How to approach the task
4. **Patterns** - Common solutions and templates
5. **Examples** - Concrete demonstrations (2-4 diverse examples)
6. **Anti-Patterns** - What to avoid and why
7. **Process** - Step-by-step workflow if applicable
8. **Validation** - Checklist or success criteria

**Adjust based on skill type:**
- **Technical skills** (coding, APIs): Emphasize patterns and examples
- **Review skills** (security, code review): Emphasize criteria and anti-patterns
- **Creation skills** (documentation, design): Emphasize process and templates

### Step 4: Write Clear Guidelines

**Format:**
- Use headings to organize concepts
- Use bullet points for lists of independent items
- Use numbered lists for sequential steps
- Use tables for comparisons or structured data
- Use code blocks for examples
- Use blockquotes for key principles or quotes

**Style:**
- Be concise but complete
- Use active voice ("Check for SQL injection" not "SQL injection should be checked")
- Provide rationale ("Why this matters")
- Include examples for ambiguous concepts

### Step 5: Add Examples and Templates

**Example Types:**

**Before/After Examples:**
```markdown
## Example: Secure Password Handling

**❌ Insecure:**
\`\`\`python
password = request.form['password']
user.password = password  # Plain text storage!
\`\`\`

**✅ Secure:**
\`\`\`python
from werkzeug.security import generate_password_hash

password = request.form['password']
user.password_hash = generate_password_hash(password)
\`\`\`
```

**Good vs. Bad Examples:**
```markdown
**Good - Parameterized Query:**
\`\`\`python
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
\`\`\`

**Bad - String Concatenation (SQL Injection Risk):**
\`\`\`python
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
\`\`\`
```

**Templates:**
```markdown
## Template: Security Review Comment

**Vulnerability:** [Type - e.g., SQL Injection, XSS]
**Severity:** [Critical | High | Medium | Low]
**Location:** [File:Line]
**Description:** [Explain the vulnerability]
**Impact:** [What could an attacker do?]
**Recommendation:** [How to fix it]
**Reference:** [OWASP link or CVE]

Example:
**Vulnerability:** SQL Injection
**Severity:** Critical
**Location:** app.py:45
**Description:** User input directly concatenated into SQL query
**Impact:** Attacker can read/modify database, potentially gain admin access
**Recommendation:** Use parameterized queries with placeholders
**Reference:** https://owasp.org/www-community/attacks/SQL_Injection
```

### Step 6: Validate and Refine

**Test your skill:**
1. Does it clearly define when to use it? (frontmatter description)
2. Can someone follow the guidelines without prior expertise?
3. Are examples concrete and realistic?
4. Is there unnecessary information that can be removed?
5. Are technical terms defined or obvious from context?

**Refine:**
- Remove redundancy
- Clarify ambiguous instructions
- Add missing examples
- Ensure consistent formatting

## Section-by-Section Guide

### Frontmatter Section

```yaml
---
name: skill-identifier-kebab-case
description: One or two sentence summary that helps Claude Code understand when to invoke this skill
---
```

**Best Practices:**
- `name`: Use kebab-case, be specific (e.g., `python-security-reviewer` not `security`)
- `description`: Focus on what the skill does and when to use it
- Keep description under 200 characters

### Introduction Section

```markdown
# Skill Title

Brief overview (2-4 sentences) explaining:
- What this skill covers
- Why it's useful
- When to apply it
```

**Example:**
```markdown
# Python Security Code Reviewer

This skill provides expert guidance for conducting security-focused code reviews of Python web applications. It covers identification of common vulnerabilities (OWASP Top 10), secure coding patterns, and actionable remediation recommendations. Use this skill when reviewing Python code for security issues before deployment or during security audits.
```

### Core Principles/Philosophy Section

**Purpose:** Establish foundational concepts

```markdown
## Core Principles

### Principle 1: [Name]
[Explanation and rationale]

### Principle 2: [Name]
[Explanation and rationale]
```

**Example:**
```markdown
## Core Security Principles

### Defense in Depth
Never rely on a single security control. Layer multiple protections (input validation + parameterized queries + least privilege DB access) so that if one fails, others provide backup.

### Fail Securely
When errors occur, fail in a secure state. Don't expose error details to users, don't grant access by default on auth failures, don't log sensitive data.

### Principle of Least Privilege
Grant minimum necessary permissions. Database users shouldn't have DROP TABLE rights. API keys should have scoped permissions. Session tokens should expire.
```

### Guidelines Section

**Purpose:** Provide actionable instructions

```markdown
## Guidelines

### [Category 1]
- [Guideline 1]
- [Guideline 2]

### [Category 2]
- [Guideline 3]
- [Guideline 4]
```

**Example:**
```markdown
## Security Review Guidelines

### Input Validation
- Validate all user inputs against allowlists, not blocklists
- Check data type, length, format, and range
- Reject invalid input; don't try to "clean" or "fix" it
- Validate on the server side even if client-side validation exists

### Authentication & Authorization
- Check that passwords are hashed (bcrypt, Argon2, PBKDF2)
- Verify session tokens are cryptographically random and expire
- Ensure authorization checks happen on every protected endpoint
- Look for privilege escalation opportunities (can user X access user Y's data?)
```

### Examples Section

**Purpose:** Demonstrate concepts concretely

Use diverse examples covering:
1. Standard successful case
2. Edge case or common mistake
3. Complex scenario if applicable

**Format:**
```markdown
## Examples

### Example 1: [Scenario Name]
[Context]

[Code or demonstration]

[Explanation]

### Example 2: [Different Scenario]
...
```

### Anti-Patterns Section

**Purpose:** Highlight common mistakes

```markdown
## Common Anti-Patterns

### ❌ Anti-Pattern 1: [Name]
**Problem:** [What's wrong]
**Example:**
\`\`\`
[Bad code]
\`\`\`
**Why it's bad:** [Explanation]
**Fix:**
\`\`\`
[Good code]
\`\`\`

### ❌ Anti-Pattern 2: [Name]
...
```

### Templates Section

**Purpose:** Provide reusable structures

```markdown
## Templates

### Template: [Use Case]
\`\`\`
[Template structure with placeholders]
\`\`\`

**Usage:**
- [Placeholder 1]: [What to fill in]
- [Placeholder 2]: [What to fill in]

**Example:**
\`\`\`
[Filled-in template]
\`\`\`
```

### Validation/Checklist Section

**Purpose:** Define success criteria

```markdown
## [Task] Checklist

Before considering the task complete, verify:
- [ ] [Criterion 1]
- [ ] [Criterion 2]
- [ ] [Criterion 3]
- [ ] [Criterion 4]

Optional but recommended:
- [ ] [Nice-to-have 1]
- [ ] [Nice-to-have 2]
```

## Domain-Specific Skill Patterns

### Technical Writing Skills (Documentation, Guides)

**Focus on:**
- Structure and organization
- Clarity and conciseness
- Target audience awareness
- Examples and templates
- Style consistency

**Key Sections:**
- Document structure guide
- Writing style guidelines
- Templates for common document types
- Examples (good vs. bad)
- Checklist for completeness

**Example Skills:**
- API documentation writer
- User manual creator
- Technical blog post writer
- README file creator

### Code-Related Skills (Languages, Frameworks)

**Focus on:**
- Idiomatic patterns
- Common pitfalls
- Best practices
- Performance considerations
- Security implications

**Key Sections:**
- Language/framework-specific patterns
- Code examples (before/after)
- Anti-patterns to avoid
- Performance tips
- Testing guidance

**Example Skills:**
- Python async programming expert
- React hooks specialist
- SQL query optimizer
- Go concurrency patterns

### Review Skills (Security, Code Quality)

**Focus on:**
- Evaluation criteria
- Common issues to look for
- Severity classification
- Remediation guidance
- Reporting format

**Key Sections:**
- Review checklist
- Issue categories and severity
- Detection patterns
- Remediation recommendations
- Report template

**Example Skills:**
- Security code reviewer
- API design reviewer
- Accessibility auditor
- Performance profiler

### Analysis Skills (Data, Metrics, Research)

**Focus on:**
- Methodology
- Data quality validation
- Analysis techniques
- Interpretation guidelines
- Visualization standards

**Key Sections:**
- Analysis process
- Data validation criteria
- Statistical methods
- Visualization best practices
- Interpretation guidelines

**Example Skills:**
- Exploratory data analyst
- A/B test evaluator
- User research synthesizer
- Performance metrics analyzer

### Design Skills (Architecture, UI/UX, Systems)

**Focus on:**
- Design principles
- Decision frameworks
- Trade-off analysis
- Pattern selection
- Validation criteria

**Key Sections:**
- Design principles
- Common patterns and when to use them
- Trade-off analysis
- Design validation
- Documentation format

**Example Skills:**
- System architecture designer
- Database schema designer
- API interface designer
- Component library designer

## Common Pitfalls to Avoid

### ❌ Too Broad Scope
**Problem:** Skill tries to cover too much
```yaml
name: developer-helper
description: Help with all development tasks
```
**Fix:** Narrow to specific expertise
```yaml
name: rest-api-versioning-guide
description: Expert guidance on API versioning strategies for RESTful services
```

### ❌ Vague Instructions
**Problem:** Guidelines are too general
```markdown
- Write good code
- Follow best practices
- Make it secure
```
**Fix:** Be specific and actionable
```markdown
- Use parameterized queries for all database operations
- Hash passwords with bcrypt (cost factor 12+)
- Validate all user inputs against allowlists before processing
```

### ❌ No Examples
**Problem:** Only abstract descriptions, no concrete demonstrations
**Fix:** Include 2-4 diverse, realistic examples

### ❌ Missing Context
**Problem:** Assumes too much prior knowledge
**Fix:** Define key terms and provide necessary background

### ❌ No Validation Criteria
**Problem:** Unclear when the task is "done"
**Fix:** Include explicit checklist or success criteria

### ❌ Redundant with General Knowledge
**Problem:** Skill just repeats common information
**Fix:** Focus on specialized, non-obvious expertise

## Skill Organization Patterns

### Single-File Skill (Simple)
```
skills/skill-name/
└── SKILL.md
```
Use when: Skill is self-contained and under ~500 lines

### Multi-File Skill (Complex)
```
skills/skill-name/
├── SKILL.md              # Main definition
├── examples.md           # Extended examples
├── templates/            # Reusable templates
│   ├── template1.md
│   └── template2.md
└── references.md         # Links and citations
```
Use when: Skill has extensive examples or multiple templates

### Skill Suite (Related Skills)
```
skills/api-design/
├── rest-api-designer/
│   └── SKILL.md
├── graphql-api-designer/
│   └── SKILL.md
└── api-documentation/
    └── SKILL.md
```
Use when: Multiple related but distinct skills in same domain

## Iterative Skill Development

Skills should evolve based on usage:

### Version 1: Minimal
- Core purpose and scope
- Basic guidelines
- 1-2 simple examples
- Basic checklist

### Version 2: Enhanced (after initial use)
- Add examples based on real usage
- Clarify ambiguous instructions
- Add anti-patterns discovered
- Expand edge cases

### Version 3: Refined (after extensive use)
- Remove redundant content
- Reorganize based on what's most-used
- Add advanced patterns
- Include references and citations

**Don't over-engineer version 1.** Start simple, then enhance based on actual needs.

## Validation Checklist

Before finalizing a new skill, verify:

**Scope & Purpose:**
- [ ] Clear, focused scope (not too broad)
- [ ] Specific expertise provided (not general knowledge)
- [ ] frontmatter description accurately summarizes the skill

**Content Quality:**
- [ ] Instructions are actionable and specific
- [ ] Examples are concrete and realistic
- [ ] Key terms are defined or obvious
- [ ] No unnecessary redundancy

**Structure:**
- [ ] Logical organization with clear sections
- [ ] Consistent formatting throughout
- [ ] Proper Markdown syntax (headings, lists, code blocks)

**Completeness:**
- [ ] Core principles or philosophy explained
- [ ] Guidelines for common scenarios provided
- [ ] At least 2 diverse examples included
- [ ] Success criteria or validation checklist present

**Usability:**
- [ ] Can be followed without prior expertise
- [ ] Examples clarify ambiguous points
- [ ] Templates provided for common outputs (if applicable)
- [ ] Clear indication of when to use this skill

## Template for New Skills

See `skill-template.md` in this directory for a blank template to start from.

## Examples of Real Skills

See `examples.md` for complete examples across different domains:
- API documentation writer
- Python security reviewer
- Data visualization designer
- Test case generator

## Using This Skill

When you need to create a new skill:

1. **Define scope** - What specific task or domain?
2. **Use the template** - Start with `skill-template.md`
3. **Gather knowledge** - Collect principles, patterns, examples
4. **Write sections** - Follow the section-by-section guide
5. **Add examples** - Include 2-4 diverse, concrete examples
6. **Validate** - Check against the validation checklist
7. **Iterate** - Start minimal, enhance based on usage

Remember: The goal is to capture specialized expertise in a reusable, actionable format. Focus on what makes this skill unique and valuable.
