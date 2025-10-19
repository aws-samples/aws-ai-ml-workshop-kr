# Skill Template

Use this template as a starting point for creating new Claude Code skills. Replace placeholders with your content and remove sections that aren't applicable.

---

```markdown
---
name: your-skill-name-in-kebab-case
description: One or two sentence description of what this skill does and when to use it
---

# [Skill Title]

[Brief introduction: 2-4 sentences explaining what this skill covers, why it's useful, and when to apply it]

## Core Principles

[If applicable: Fundamental concepts or philosophy underlying this skill]

### Principle 1: [Name]
[Explanation and rationale]

### Principle 2: [Name]
[Explanation and rationale]

## Guidelines

[Actionable instructions for performing the task]

### [Category 1]
- [Guideline 1]
- [Guideline 2]
- [Guideline 3]

### [Category 2]
- [Guideline 4]
- [Guideline 5]

### [Category 3]
- When [condition], do [action]
- When [condition], do [action]

## Patterns

[Common solutions, approaches, or templates for recurring scenarios]

### Pattern 1: [Name]
**When to use:** [Condition or scenario]

**Structure:**
\`\`\`
[Code, template, or structure]
\`\`\`

**Example:**
\`\`\`
[Concrete example]
\`\`\`

### Pattern 2: [Name]
[Similar structure]

## Examples

[2-4 diverse, concrete examples demonstrating the skill in action]

### Example 1: [Standard Case]
[Context and description]

\`\`\`
[Code, text, or demonstration]
\`\`\`

[Explanation of why this is good]

### Example 2: [Edge Case or Common Mistake]
**❌ Problematic approach:**
\`\`\`
[What not to do]
\`\`\`

**✅ Better approach:**
\`\`\`
[What to do instead]
\`\`\`

[Explanation]

### Example 3: [Complex Scenario]
[More advanced or nuanced example]

## Anti-Patterns

[Common mistakes to avoid]

### ❌ Anti-Pattern 1: [Name]
**Problem:** [What's wrong with this approach]

**Example:**
\`\`\`
[Demonstration of the anti-pattern]
\`\`\`

**Why it's problematic:** [Explanation of consequences]

**Better approach:**
\`\`\`
[Correct way to do it]
\`\`\`

### ❌ Anti-Pattern 2: [Name]
[Similar structure]

## Process

[If applicable: Step-by-step workflow for complex tasks]

**Step 1: [Action]**
- [Sub-task]
- [Sub-task]

**Step 2: [Action]**
- [Sub-task]
- [Sub-task]

**Step 3: [Action]**
- [Sub-task]

## Templates

[If applicable: Reusable structures or formats]

### Template: [Use Case Name]

\`\`\`
[Template with placeholders like [PLACEHOLDER]]
\`\`\`

**How to use:**
- `[PLACEHOLDER1]`: [What to fill in here]
- `[PLACEHOLDER2]`: [What to fill in here]

**Example:**
\`\`\`
[Filled template showing actual usage]
\`\`\`

## Validation Checklist

Before considering the task complete, verify:

**Required:**
- [ ] [Critical criterion 1]
- [ ] [Critical criterion 2]
- [ ] [Critical criterion 3]
- [ ] [Critical criterion 4]

**Recommended:**
- [ ] [Nice-to-have 1]
- [ ] [Nice-to-have 2]

## References

[If applicable: Links to authoritative sources, standards, or documentation]

- [Resource 1]: [URL or description]
- [Resource 2]: [URL or description]

---

## Usage Notes

[Optional: Any additional context, tips, or guidance for using this skill effectively]
```

---

## Template Usage Instructions

1. **Copy this template** to a new file: `skills/your-skill-name/SKILL.md`

2. **Fill in the frontmatter:**
   - Choose a descriptive, kebab-case name
   - Write a clear 1-2 sentence description

3. **Replace all placeholders** in square brackets `[like this]`

4. **Remove sections** that don't apply to your skill (but keep at least: Introduction, Guidelines, Examples, Validation)

5. **Add concrete examples** - This is crucial! Show, don't just tell.

6. **Keep it focused** - If a skill file grows beyond ~800 lines, consider splitting it or using supporting files (examples.md, templates/)

7. **Validate** using the checklist from the main SKILL.md

## Quick Start: Minimal Skill

If you want to start with the absolute minimum structure:

```markdown
---
name: skill-name
description: What this skill does
---

# Skill Title

[Brief introduction]

## Guidelines

- [Guideline 1]
- [Guideline 2]
- [Guideline 3]

## Examples

### Example 1
[Concrete example]

### Example 2
[Another example]

## Checklist

- [ ] [Success criterion 1]
- [ ] [Success criterion 2]
```

Then expand from there as needed.
