# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Validator Agent Integration Plan

**IMPORTANT**: The Reference Validator Agent system is already VALIDATED and WORKING. This plan is to integrate the proven components from the Reference system into the Target system with minimal modifications.

### Project Goal

Integrate the **validated** Validator Agent system from Reference system (`/home/ubuntu/Self-Study-Generative-AI/lab/11_bedrock_manus`) into Target system while maintaining existing code structure.

### Reference System Components (ALREADY VALIDATED)

**Core Validated Files**:
1. `/home/ubuntu/Self-Study-Generative-AI/lab/11_bedrock_manus/src/tools/validator_tools.py`
   - `OptimizedValidator` class for performance optimization
   - Priority-based validation (high/medium/low importance)
   - Batch processing and data caching
   - Tool specifications for validator agent

2. `/home/ubuntu/Self-Study-Generative-AI/lab/11_bedrock_manus/src/tools/calculation_tracker.py`
   - `CalculationTracker` class for automatic metadata collection
   - Decorators for calculation tracking
   - Manual tracking functions

3. `/home/ubuntu/Self-Study-Generative-AI/lab/11_bedrock_manus/src/prompts/validator.md`
   - Complete validator agent prompt template
   - Performance optimization instructions
   - Citation generation workflow

4. `/home/ubuntu/Self-Study-Generative-AI/lab/11_bedrock_manus/src/graph/nodes.py:validator_node`
   - Working LangGraph validator node implementation
   - State management and command structure

### Target System Integration Strategy

**Key Requirement**: Transform Reference LangGraph `validator_node` into Target Strands `validator_agent_tool`

## Code Generation Plan

### 1. Create `src/tools/calculation_tracker.py`
- **Source**: Copy from Reference system with minimal adaptation
- **Purpose**: Automatic calculation metadata tracking
- **Changes**: Adapt import paths for Target system

### 2. Create `src/tools/validator_agent_tool.py` 
- **Source**: Adapt Reference `validator_node` logic into Strands tool pattern
- **Pattern**: Follow `src/tools/reporter_agent_tool.py` exactly
- **Key Adaptations**:
  - Transform LangGraph Command structure to Strands ToolResult
  - Adapt state management from LangGraph State to `_global_node_states`
  - Maintain streaming response and observability patterns

### 3. Create `src/prompts/validator.md`
- **Source**: Copy Reference validator prompt with TARGET system adaptations
- **Changes**: Update file paths and tool references for Target system

### 4. Update `src/graph/nodes.py:supervisor_node`
- **Change**: Add `validator_agent_tool` to tools list
- **Pattern**: `tools=[coder_agent_tool, reporter_agent_tool, tracker_agent_tool, validator_agent_tool]`

### 5. Update `src/config/tools.py`
- **Change**: Register `validator_agent_tool` following existing patterns

### 6. Enhance `src/prompts/coder.md` (Additive only)
- **Add**: Calculation tracking instructions from Reference system
- **Include**: 
  - Importance classification (high/medium/low)
  - Metadata recording after calculations
  - `calculation_tracker` integration

### 7. Enhance `src/prompts/reporter.md` (Additive only)
- **Add**: Citation usage instructions
- **Include**:
  - Read `./artifacts/citations.json` 
  - Citation formatting in reports
  - Validation result references

## Implementation Requirements

### Code Generation Priorities
1. **COPY Reference components** - Don't reinvent, use validated code
2. **ADAPT architecture patterns** - LangGraph → Strands transformation
3. **PRESERVE Target structure** - No modification to existing functions
4. **MINIMAL changes** - Only essential adaptations for compatibility

### Critical Adaptations Needed

**From Reference LangGraph to Target Strands**:
```python
# Reference pattern (LangGraph):
def validator_node(state: State) -> Command[Literal["supervisor"]]:
    # ... validation logic ...
    return Command(update={...}, goto="supervisor")

# Target pattern (Strands Tool):
def validator_agent_tool(tool: ToolUse, **_kwargs: Any) -> ToolResult:
    # ... same validation logic ...
    return {"toolUseId": tool_use_id, "status": "success", "content": [...]}
```

**State Management Adaptation**:
```python
# Reference: LangGraph state
state.get("messages", [])

# Target: Global state
shared_state = _global_node_states.get('shared', {})
shared_state.get('messages', [])
```

### File Structure (Target System)
```
src/
├── tools/
│   ├── calculation_tracker.py          # NEW: Copy from Reference
│   └── validator_agent_tool.py         # NEW: Adapt from Reference validator_node
├── prompts/
│   ├── validator.md                     # NEW: Copy from Reference
│   ├── coder.md                        # MODIFY: Add tracking instructions
│   └── reporter.md                     # MODIFY: Add citation instructions
├── graph/
│   └── nodes.py                        # MODIFY: Add validator tool to supervisor
└── config/
    └── tools.py                        # MODIFY: Register validator tool
```

### Validation Workflow Integration

**Target System Flow**:
1. Coordinator → Planner → Supervisor (unchanged)
2. Supervisor calls `validator_agent_tool` when needed
3. Validator agent validates calculations and generates citations
4. Reporter agent uses citations for final report
5. All artifacts saved to `./artifacts/`

### Success Criteria
- [ ] `calculation_tracker.py` copied and adapted
- [ ] `validator_agent_tool.py` created following Strands pattern
- [ ] `validator.md` prompt copied and adapted
- [ ] Supervisor node updated with validator tool
- [ ] Tool registration completed
- [ ] Existing prompts enhanced with validator instructions
- [ ] End-to-end validation workflow functioning
- [ ] Citation generation and usage working
- [ ] No existing functionality broken

## Key Implementation Notes

1. **REUSE validated code** - Reference system is working, don't recreate
2. **FOLLOW Target patterns** - Use `reporter_agent_tool.py` as template
3. **MAINTAIN compatibility** - All existing features must continue working
4. **PRESERVE performance** - Keep Reference optimization (priority-based, batch processing)
5. **MINIMAL footprint** - Only necessary changes, maximum code reuse

### Expected Output Files
- `./artifacts/calculation_metadata.json` - From coder (via calculation_tracker)
- `./artifacts/citations.json` - From validator agent
- `./artifacts/validation_report.txt` - Validation summary
- Enhanced reports with numerical citations

## Testing

### Environment Setup

**IMPORTANT**: Always use the project's virtual environment for testing and development:

```bash
# Activate virtual environment
source setup/.venv/bin/activate

# Run unit tests as needed
python -m pytest tests/ -v  # if test directory exists

# MANDATORY: Always test template.py after any prompt modifications
python3 template.py

### Component Unit Testing

#### Korean Font Testing
```bash
# Test Korean font setup in coder.md
source setup/.venv/bin/activate && python3 -c "
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# Test Korean font setup from coder.md
plt.rcParams['font.family'] = ['NanumGothic']
plt.rcParams['font.sans-serif'] = ['NanumGothic', 'NanumBarunGothic', 'NanumMyeongjo', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False

korean_font_prop = fm.FontProperties(family='NanumGothic')

# Create simple test chart
fig, ax = plt.subplots(figsize=(4, 2))
ax.text(0.5, 0.5, '한글폰트테스트', fontproperties=korean_font_prop, ha='center', va='center')
ax.set_title('폰트 확인', fontproperties=korean_font_prop)

os.makedirs('./artifacts', exist_ok=True)
plt.savefig('./artifacts/font_unit_test.png', dpi=100, bbox_inches='tight')
plt.close()
print('✅ Korean font unit test completed: ./artifacts/font_unit_test.png')
"
```

#### PDF Generation Testing  
```bash
# Test reporter.md PDF generation capabilities
source setup/.venv/bin/activate && python3 -c "
import weasyprint
import markdown
print('✅ WeasyPrint available for PDF generation')
print('✅ Markdown parser available')
print('✅ Reporter PDF generation dependencies OK')
"
```

## Key Code Paths

### Target Code Path (Implementation Location)
```
/home/ubuntu/aws-ai-ml-workshop-kr/genai/aws-gen-ai-kr/20_applications/08_bedrock_manus/use_cases/05_insight_extractor_strands_sdk_workshop_phase_2
```

### Reference Code Path (Source of Validated Components)
```
/home/ubuntu/Self-Study-Generative-AI/lab/11_bedrock_manus
```
### Backup of Original Target Code Path 
```
/home/ubuntu/05_insight_extractor_strands_sdk_workshop_phase_2
```

### File Mapping (Reference → Target)
```
Reference System                                    →  Target System
─────────────────────────────────────────────────────────────────────────────────
/home/ubuntu/Self-Study-Generative-AI/             →  /home/ubuntu/aws-ai-ml-workshop-kr/genai/aws-gen-ai-kr/20_applications/08_bedrock_manus/use_cases/
lab/11_bedrock_manus/src/tools/                    →  05_insight_extractor_strands_sdk_workshop_phase_2/src/tools/
├── validator_tools.py                             →  ├── validator_agent_tool.py (ADAPT)
└── calculation_tracker.py                         →  └── calculation_tracker.py (COPY)

/home/ubuntu/Self-Study-Generative-AI/             →  /home/ubuntu/aws-ai-ml-workshop-kr/genai/aws-gen-ai-kr/20_applications/08_bedrock_manus/use_cases/
lab/11_bedrock_manus/src/prompts/                  →  05_insight_extractor_strands_sdk_workshop_phase_2/src/prompts/
├── validator.md                                   →  ├── validator.md (COPY + ADAPT)
├── coder.md (validator sections)                  →  ├── coder.md (ADD validator sections)
└── reporter.md (citation sections)                →  └── reporter.md (ADD citation sections)

/home/ubuntu/Self-Study-Generative-AI/             →  /home/ubuntu/aws-ai-ml-workshop-kr/genai/aws-gen-ai-kr/20_applications/08_bedrock_manus/use_cases/
lab/11_bedrock_manus/src/graph/nodes.py            →  05_insight_extractor_strands_sdk_workshop_phase_2/src/graph/nodes.py
└── validator_node() function                      →  └── supervisor_node() - add validator_agent_tool to tools list
```

### Implementation Priority Order
1. **COPY**: `calculation_tracker.py` (minimal changes)
2. **ADAPT**: `validator_agent_tool.py` (LangGraph → Strands pattern)
3. **COPY+ADAPT**: `validator.md` prompt
4. **UPDATE**: `supervisor_node` tools list
5. **REGISTER**: Tool in `config/tools.py`
6. **ENHANCE**: `coder.md` and `reporter.md` prompts