# Section Organization Best Practices

This guide provides detailed recommendations for organizing system prompts into distinct sections using the **Hybrid (Markdown + XML) approach**.

**Part of the system-prompt-writer skill** - See `SKILL.md` for core principles and broader context.

## When to Use This Guide

- **Structuring a complex prompt?** This guide shows how to organize multiple sections
- **Choosing section names?** See recommended section types and naming conventions
- **Ordering sections?** Find optimal section sequencing patterns
- **Need detailed examples?** See section-by-section breakdowns

## Related Files

- **SKILL.md** - Core context engineering principles and guidelines
- **examples.md** - Complete real-world examples showing these patterns in action
- **skill-template.md** - Ready-to-use templates with proper section structure

## Why Section Organization Matters

**Anthropic's Recommendation:**
> "We recommend organizing prompts into distinct sections (like `<background_information>`, `<instructions>`, `## Tool guidance`, `## Output description`, etc) and using techniques like XML tagging or Markdown headers to delineate these sections."

**Benefits:**
- **Clarity**: Clear boundaries between different types of information
- **Maintainability**: Easy to update specific sections without affecting others
- **Parseability**: Helps LLMs distinguish between context, instructions, and constraints
- **Progressive Disclosure**: Enables loading only relevant sections as needed
- **Debugging**: Easier to identify which section causes issues

## Recommended Format: Hybrid (Markdown + XML)

Use **Markdown headers** for section structure and **XML tags** for content:

```markdown
## Section Name
<section_tag>
Content goes here
</section_tag>
```

**Why Hybrid?**
- ✅ Human-readable (Markdown headers)
- ✅ Machine-parseable (XML tags)
- ✅ Matches Anthropic's examples
- ✅ Best of both worlds

## Recommended Section Types

### 1. Role Definition
**Purpose:** Establishes agent identity and primary objective

**Section Format:**
```markdown
## Role
<role>
[Content]
</role>
```

**What to Include:**
- Who the agent is (job title, expertise)
- Primary objective or goal
- Key responsibilities (2-3 bullets max)

**Example:**
```markdown
## Role
<role>
You are a customer service specialist for TechCorp. Your objective is to resolve customer inquiries efficiently while maintaining a positive experience.
</role>
```

### 2. Background Information
**Purpose:** Provides context that informs decision-making

**Section Names:**
- `<background_information>` or `## Background Information`
- `<context>` or `## Context`
- `<domain_knowledge>` or `## Domain Knowledge`

**What to Include:**
- Essential domain knowledge
- Business rules or policies
- Assumptions the agent should make
- Relevant constraints from the environment

**What to EXCLUDE:**
- Historical information that doesn't affect decisions
- Redundant information available elsewhere
- Generic background (focus on actionable context)

**Example:**
```xml
<background_information>
TechCorp sells SaaS products with 3 pricing tiers: Basic ($10/mo), Pro ($50/mo), Enterprise (custom).
Support hours are 9am-5pm EST Monday-Friday.
Enterprise customers have priority support with 1-hour SLA.
</background_information>
```

### 3. Instructions / Guidelines
**Purpose:** Directs how the agent should behave and make decisions

**Section Names:**
- `<instructions>` or `## Instructions`
- `<guidelines>` or `## Guidelines`
- `<behavior>` or `## Behavior`

**What to Include:**
- Step-by-step workflows (only when necessary)
- Decision-making principles
- When-then rules for specific scenarios
- Priority ordering if conflicts arise

**Formatting Tips:**
- Use bullet points for independent guidelines
- Use numbered lists for sequential steps
- Use "When X, do Y" format for conditional logic

**Example:**
```xml
<instructions>
- Greet customers warmly and identify their account status
- Listen actively and confirm understanding before providing solutions
- When customer is upset: acknowledge frustration, apologize, focus on resolution
- When issue requires escalation: explain clearly and set expectations
- Always end by asking if anything else is needed
</instructions>
```

### 4. Tool Guidance
**Purpose:** Specifies when and how to use available tools

**Section Names:**
- `<tool_guidance>` or `## Tool Guidance`
- `<tools>` or `## Tools`
- `<available_actions>` or `## Available Actions`

**What to Include:**
- List of available tools
- Specific conditions for using each tool
- Expected inputs and outputs
- Decision tree for tool selection

**Key Principle:**
> "If a human engineer can't definitively say which tool should be used in a given situation, an AI agent can't be expected to do better."

**Example:**
```xml
<tool_guidance>
Tool Selection:
- search_order(order_id): Use when customer asks about order status
- process_refund(order_id, reason): Use for returns within 30 days
- escalate_to_specialist(ticket): Use for technical issues or billing disputes
- update_account(customer_id, fields): Use for account information changes

Decision Tree:
1. Order status question → search_order
2. Return request + within 30 days → process_refund
3. Return request + after 30 days → escalate_to_specialist
4. Technical problem → escalate_to_specialist
5. Account update (email, address) → update_account
</tool_guidance>
```

### 5. Output Format
**Purpose:** Defines expected structure and format of responses

**Section Names:**
- `<output_format>` or `## Output Format`
- `<response_format>` or `## Response Format`
- `<output_description>` or `## Output Description`

**What to Include:**
- Required structure of responses
- Format specifications (JSON, Markdown, plain text)
- Examples of well-formatted outputs
- Style and tone guidelines

**Example:**
```xml
<output_format>
Response Structure:
1. Greeting and acknowledgment
2. Direct answer or solution
3. Additional relevant information (if applicable)
4. Closing and offer for further assistance

Tone: Friendly, professional, empathetic
Length: Concise (2-4 sentences for simple queries)
Formatting: Use bullet points for lists, bold for emphasis
</output_format>
```

### 6. Success Criteria
**Purpose:** Defines what constitutes task completion

**Section Names:**
- `<success_criteria>` or `## Success Criteria`
- `<completion_criteria>` or `## Completion Criteria`
- `<quality_standards>` or `## Quality Standards`

**What to Include:**
- Measurable indicators of success
- Quality standards to meet
- When the agent should consider the task complete
- Edge cases that still count as success

**Example:**
```xml
<success_criteria>
Task is complete when:
- Customer's question is fully answered
- Appropriate tool(s) were used correctly
- Response is accurate and helpful
- Follow-up offer was made
- Tone was professional and empathetic

It's acceptable to:
- Ask clarifying questions if request is ambiguous
- Escalate if issue is beyond your capability
- Admit uncertainty rather than guess
</success_criteria>
```

### 7. Constraints
**Purpose:** Establishes boundaries and limitations

**Section Names:**
- `<constraints>` or `## Constraints`
- `<limitations>` or `## Limitations`
- `<boundaries>` or `## Boundaries`

**What to Include:**
- What the agent must NOT do
- Privacy and security requirements
- Scope limitations
- Required confirmations or approvals

**Example:**
```xml
<constraints>
Do NOT:
- Process refunds over $500 without manager approval
- Share customer information across accounts
- Make promises about future product features
- Bypass security verification procedures

Always:
- Verify customer identity before accessing account details
- Respect data privacy regulations
- Stay within defined scope (customer support only)
- Document all escalations
</constraints>
```

### 8. Examples (Optional but Highly Recommended)
**Purpose:** Demonstrates desired behavior through concrete examples

**Anthropic's Principle:**
> "For an LLM, examples are the 'pictures' worth a thousand words."

**Section Names:**
- `<examples>` or `## Examples`
- `<sample_interactions>` or `## Sample Interactions`

**What to Include:**
- 2-3 **diverse, canonical examples** that portray expected behavior
- Examples demonstrating the range of scenarios (not every edge case)
- Both successful outputs AND edge case handling
- Examples that show tool usage in context (if applicable)

**What NOT to Include:**
- A "laundry list of edge cases" attempting to cover every possible scenario
- Redundant or overlapping examples
- Dozens of examples for minor variations

**When to Use:**
- Complex or ambiguous tasks
- When output format is non-standard
- When specific phrasing is important
- To replace lengthy instructions (examples can show what paragraphs struggle to tell)

**Best Practices:**
1. **Curate carefully** - Choose examples that teach behavior patterns, not memorize responses
2. **Show diversity** - Cover different types of interactions (standard, edge case, tool usage)
3. **Keep it minimal** - 2-3 good examples beat 20 mediocre ones
4. **Demonstrate, don't enumerate** - Examples should illustrate principles, not list rules

**Example:**
```xml
<examples>
Example 1: Order Status Query
User: "Where's my order #12345?"
Agent: "I've looked up order #12345. It shipped on Oct 15 and is currently in transit. Expected delivery is Oct 19. You can track it here: [link]. Anything else I can help with?"

Example 2: Refund Request (Within Policy)
User: "I want to return this, I ordered it 2 weeks ago"
Agent: "I can help with that return. I've started the refund process for your order. You'll receive a prepaid shipping label within 24 hours via email. Once we receive the item, your refund will be processed in 3-5 business days. Is there anything else?"

Example 3: Escalation Needed
User: "Your software keeps crashing when I export reports"
Agent: "I'm sorry you're experiencing crashes during export. This sounds like a technical issue that requires our engineering team. I'm creating a high-priority ticket and our specialist will contact you within 4 hours. Can I get your best contact email and phone number?"
</examples>
```

## Section Organization Patterns

All patterns use the **Hybrid (Markdown + XML) format**.

### Pattern 1: Simple Agent (Minimal Sections)

```markdown
## Role
<role>
[Who and what]
</role>

## Instructions
<instructions>
[How to behave]
</instructions>

## Constraints
<constraints>
[What not to do]
</constraints>
```

**Use for:** Single-purpose agents, simple tasks, minimal tools

### Pattern 2: Standard Agent (Recommended)

```markdown
## Role
<role>
[Who and what]
</role>

## Background Information
<background_information>
[Relevant context]
</background_information>

## Instructions
<instructions>
[How to behave]
</instructions>

## Tool Guidance
<tool_guidance>
[When to use which tools]
</tool_guidance>

## Success Criteria
<success_criteria>
[What constitutes completion]
</success_criteria>

## Constraints
<constraints>
[What not to do]
</constraints>
```

**Use for:** Most production agents, moderate complexity

### Pattern 3: Complex Agent (Full Specification)

```markdown
## Role
<role>
[Who and what]
</role>

## Background Information
<background_information>
[Relevant context]
</background_information>

## Instructions
<instructions>
[How to behave]
</instructions>

## Tool Guidance
<tool_guidance>
[When to use which tools]
</tool_guidance>

## Decision Framework
<decision_framework>
[How to make choices]
</decision_framework>

## Success Criteria
<success_criteria>
[What constitutes completion]
</success_criteria>

## Constraints
<constraints>
[What not to do]
</constraints>

## Examples
<examples>
[Sample interactions - 2-3 diverse canonical examples]
</examples>
```

**Use for:** Multi-step workflows, high-stakes applications, complex decision-making

### Pattern 4: Multi-Agent System

```markdown
## Agent Identity
Name: [agent_name]
Type: [coordinator|specialist|worker]

## Role in System
[How this agent fits in the larger system]

## Communication Protocol
Input format: [what to expect from other agents]
Output format: [what to send to other agents]

## Handoff Criteria
[When to delegate to other agents]

## Instructions
[Agent-specific behavior]

## Tool Guidance
[Available tools and usage]
```

**Use for:** Agent networks, hierarchical systems, specialized sub-agents

## Ordering Guidelines

### Recommended Order (General → Specific)

1. **Role** - Start with identity and purpose
2. **Background Information** - Provide essential context
3. **Instructions** - Explain how to behave
4. **Tool Guidance** - Detail when to use tools
5. **Output Format** - Specify response structure
6. **Success Criteria** - Define completion
7. **Constraints** - Establish boundaries
8. **Examples** - Demonstrate concretely (if needed)

### Alternative Order (What → How → Why → Boundaries)

1. **Role** - What the agent is
2. **Capabilities** - What the agent can do
3. **Instructions** - How the agent should act
4. **Success Criteria** - Why actions matter
5. **Constraints** - Boundaries on behavior

**Choose based on:** Complexity, audience, agent type

**Note:** All patterns use the Hybrid (Markdown + XML) format as recommended. This provides the best combination of human readability and machine parseability.

## Common Mistakes

❌ **Too many sections** - Creates unnecessary complexity
✅ **Just enough sections** - Balance structure with simplicity

❌ **Vague section names** - "Information", "Details", "Other"
✅ **Descriptive section names** - "Tool Guidance", "Success Criteria"

❌ **Sections with unrelated content** - Mixing instructions and constraints
✅ **Focused sections** - Each section has a clear, single purpose

❌ **Redundancy across sections** - Repeating the same information
✅ **DRY principle** - Each piece of information appears once

❌ **Missing critical sections** - No success criteria or constraints
✅ **Complete coverage** - All necessary information present

## Section Organization Checklist

Before finalizing your prompt, verify:

- [ ] Each section has a clear, specific purpose
- [ ] Section names are descriptive and consistent
- [ ] Sections are ordered logically (general → specific)
- [ ] No redundancy between sections
- [ ] All critical information is present
- [ ] Sections use appropriate formatting (XML vs Markdown)
- [ ] Total length is minimized (cut unnecessary sections)
- [ ] Sections support progressive disclosure if needed

## Real-World Example: Before & After

### Before (Unstructured)

```
You are a helpful customer service agent for TechCorp, a SaaS company founded in 2010 by Jane Smith and John Doe. We value customer satisfaction and innovation. Be friendly and professional. Answer customer questions about orders, refunds, and technical issues. Use the search_order tool to look up orders. Don't share customer data. Process refunds if within 30 days. Escalate technical issues. Our support hours are 9am-5pm EST. Enterprise customers get priority. Be empathetic when customers are upset. Always offer additional help before ending conversation.
```

**Issues:**
- No clear structure
- Mixed information types
- Hard to update or maintain
- Difficult to parse programmatically

### After (Well-Structured)

```markdown
## Role
<role>
You are a customer service specialist for TechCorp, a SaaS company. Your objective is to resolve customer inquiries efficiently while maintaining satisfaction.
</role>

## Background Information
<background_information>
- Support hours: 9am-5pm EST Monday-Friday
- Enterprise customers have priority support
- Standard refund window: 30 days
</background_information>

## Instructions
<instructions>
- Greet customers professionally and warmly
- When customers are upset: acknowledge, empathize, focus on solution
- Always offer additional help before ending conversation
- Escalate technical issues to specialists
</instructions>

## Tool Guidance
<tool_guidance>
- search_order(order_id): Use for order status inquiries
- process_refund(order_id, reason): Use for returns within 30 days
- escalate_ticket(description, priority): Use for technical issues
</tool_guidance>

## Constraints
<constraints>
- Never share customer data across accounts
- Do not process refunds outside 30-day window without approval
- Stay within customer support scope (no engineering decisions)
</constraints>
```

**Improvements:**
- Clear section boundaries
- Easy to update specific parts
- Grouped related information
- Removed non-essential background
- Ready for progressive disclosure

---

**Remember:** Section organization is a means to an end - clarity and effectiveness. Don't over-structure for its own sake. Use just enough organization to make your prompt clear, maintainable, and effective.

## Next Steps

After learning about section organization:

1. **Apply to real prompts** - See `examples.md` for complete prompts using these patterns
2. **Use templates** - Start with `skill-template.md` which already has proper section structure
3. **Understand principles** - Read `SKILL.md` for why these structures work (context engineering)
4. **Validate your work** - Use the checklist in `SKILL.md` to ensure proper organization

## Quick Reference

| Need | Go To |
|------|-------|
| Section structure patterns | This file (section-organization-guide.md) |
| Complete working examples | examples.md |
| Blank templates to start from | skill-template.md |
| Core principles and validation | SKILL.md |

---

**Return to main skill:** See `SKILL.md` for the complete system-prompt-writer skill.
