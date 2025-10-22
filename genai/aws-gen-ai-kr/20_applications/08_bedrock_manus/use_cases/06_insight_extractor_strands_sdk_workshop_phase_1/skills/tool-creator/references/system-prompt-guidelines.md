# System Prompt Guidelines for Agent Tools

This document provides guidelines for creating effective system prompts for agent-as-a-tool implementations. It references the comprehensive `system-prompt-writer` skill with key points specific to tool creation.

## Quick Reference

For complete system prompt writing guidance, refer to `/skills/system-prompt-writer/SKILL.md`.

This document focuses on the most critical aspects for agent tool prompts.

## Essential Principles for Agent Tool Prompts

### 1. Template Variable Escaping (CRITICAL)

**This is the #1 cause of agent tool failures.**

The project uses a template system (`src/prompts/template.py`) that processes prompts with `.format()`. You MUST follow escaping rules:

**Escaping Rule:**
- **Single braces `{}`** → Template variables (e.g., `{USER_REQUEST}`, `{FULL_PLAN}`)
- **Double braces `{{}}`** → Escaped to single braces in output (for code samples)

**Common Mistakes:**

❌ **WRONG** (Will cause KeyError):
```python
result = {"key": "value"}
print(f"Total: {amount}")
```

✅ **CORRECT**:
```python
result = {{"key": "value"}}
print(f"Total: {{amount}}")
```

**Required Template Variables for Agent Tools:**

Always include these in your prompt frontmatter:
```markdown
---
USER_REQUEST: {USER_REQUEST}
FULL_PLAN: {FULL_PLAN}
---
```

### 2. Recommended Structure for Agent Tool Prompts

Use the Hybrid (Markdown + XML) approach:

```markdown
## Role
<role>
You are a [specific role]. Your objective is to [clear goal].
</role>

## Instructions
<instructions>
- [Key principle 1]
- [Key principle 2]
- When [situation], do [action]
</instructions>

## Tool Guidance
<tool_guidance>
- tool_name: Use when [specific condition]
- tool_name_2: Use when [specific condition]
</tool_guidance>

## Constraints
<constraints>
- Do not [constraint 1]
- Always [requirement 1]
</constraints>
```

### 3. Tool Guidance - Be Specific

**Poor Tool Guidance:**
```
You have access to python_repl_tool and bash_tool. Use them as needed.
```

**Good Tool Guidance:**
```markdown
## Tool Guidance
<tool_guidance>
- python_repl_tool(code): Use for data analysis, calculations, and generating visualizations
- bash_tool(cmd): Use for file system operations, checking file existence, and listing directories

Decision Framework:
- Data manipulation or analysis → python_repl_tool
- File operations → bash_tool
</tool_guidance>
```

### 4. Domain-Specific Patterns for Common Agent Tools

#### Execution/Worker Agents (e.g., Coder)

Focus on:
- Tool usage and execution safety
- Error handling
- Output formatting
- Validation before execution

**Template:**
```markdown
## Role
<role>
You are a code execution specialist. Execute Python/bash commands safely and return results.
</role>

## Capabilities
<capabilities>
- Execute Python in REPL environment
- Run bash commands for file operations
- Handle errors gracefully
- Save artifacts to designated locations
</capabilities>

## Safety Constraints
<constraints>
- Validate inputs before execution
- Never execute potentially harmful code
- Respect file system boundaries
</constraints>
```

#### Report/Content Generation Agents (e.g., Reporter)

Focus on:
- Content structure
- Formatting standards
- Visualization integration
- Multi-format output

**Template:**
```markdown
## Role
<role>
You are a report generation specialist. Create comprehensive, well-formatted reports.
</role>

## Report Structure
<structure>
Standard sections:
1. Executive Summary
2. Findings/Analysis
3. Visualizations
4. Conclusions/Recommendations
</structure>

## Formatting Standards
<formatting>
- Use consistent heading levels
- Label all charts and tables
- Keep paragraphs concise
- Use bullet points for key findings
</formatting>
```

#### Validation/Quality Assurance Agents (e.g., Validator)

Focus on:
- Validation criteria
- Quality checks
- Error detection
- Feedback format

**Template:**
```markdown
## Role
<role>
You are a validation specialist. Verify outputs meet quality standards.
</role>

## Validation Criteria
<validation_criteria>
- Completeness: All required sections present
- Accuracy: Data and calculations correct
- Format: Proper structure and formatting
- Consistency: Style and terminology consistent
</validation_criteria>

## Validation Process
<process>
1. Check structural requirements
2. Verify data accuracy
3. Review formatting
4. Provide detailed feedback
</process>
```

#### Progress Tracking Agents (e.g., Tracker)

Focus on:
- State monitoring
- Progress reporting
- Completion tracking
- Next step identification

**Template:**
```markdown
## Role
<role>
You are a progress tracking specialist. Monitor task completion and guide next steps.
</role>

## Tracking Responsibilities
<responsibilities>
- Monitor completed tasks
- Identify remaining work
- Suggest next actions
- Report overall progress
</responsibilities>

## Progress Reporting
<reporting>
Format progress reports with:
- Completed tasks (with checkmarks)
- In-progress tasks
- Pending tasks
- Recommended next step
</reporting>
```

### 5. Minimum Effective Information

**Key Question:** What's the smallest amount of context needed for the agent to succeed?

**Before (Over-specified):**
```
You are a data analyst for the XYZ project. You should always be helpful and professional. When analyzing data, make sure to follow best practices. Use Python for calculations. Always double-check your work...
```

**After (Optimized):**
```
You are a data analyst. Execute Python code for data analysis and calculations. Validate results before returning.
```

### 6. Iterative Development

**Don't try to write the perfect prompt on the first try.**

1. Start minimal (role + basic instructions)
2. Test with real tasks
3. Identify failure modes
4. Add targeted improvements
5. Re-test

**Example Evolution:**

**Version 1 (Minimal):**
```
You are a coder agent. Execute Python and bash commands.
```

**Version 2 (After finding it doesn't validate inputs):**
```
You are a coder agent. Execute Python and bash commands.

Before execution:
- Validate code syntax
- Check for potentially destructive operations
```

**Version 3 (After finding poor error messages):**
```
You are a coder agent. Execute Python and bash commands.

Before execution:
- Validate code syntax
- Check for potentially destructive operations

Error Handling:
- Capture all errors with full stack traces
- Provide clear error messages
- Suggest fixes when possible
```

## Pre-Writing Checklist

Before creating any agent tool prompt:

- [ ] Identified agent's primary role and purpose
- [ ] Determined which tools the agent needs
- [ ] Planned template variables: `{USER_REQUEST}`, `{FULL_PLAN}`
- [ ] Will use double braces `{{}}` for ALL code samples
- [ ] Chosen appropriate domain-specific pattern
- [ ] Started with minimal prompt (will iterate)

## Common Pitfalls to Avoid

❌ **Missing brace escaping** - Causes KeyError, agent won't load
❌ **Over-specification** - Writing step-by-step algorithms instead of guidance
❌ **Vague tool guidance** - "Use tools as needed" instead of specific conditions
❌ **Redundancy** - Repeating information from tool descriptions
❌ **Premature optimization** - Writing complex prompt before testing
❌ **Missing template variables** - Not including `{USER_REQUEST}` and `{FULL_PLAN}`

## Complete Example: Coder Agent Prompt

See `references/tool-examples.md` for complete working examples.

## For More Details

This is a quick reference. For comprehensive guidance:

- **Full system prompt guide**: `/skills/system-prompt-writer/SKILL.md`
- **Examples**: `/skills/system-prompt-writer/references/examples.md`
- **Section organization**: `/skills/system-prompt-writer/references/section-organization-guide.md`
- **Real tool prompts**: `/src/prompts/coder.md`, `/src/prompts/reporter.md`, etc.

## Validation

After writing your prompt, verify:

1. **Template Variables**: Used single braces `{}` for variables, double braces `{{}}` for code
2. **Structure**: Clear sections with Markdown + XML
3. **Tool Guidance**: Specific conditions for each tool
4. **Role**: Clear and focused
5. **Constraints**: Explicit boundaries
6. **Testing**: Plan to test and iterate

Remember: **Effective > Perfect**. Start simple, test, and improve based on real behavior.
