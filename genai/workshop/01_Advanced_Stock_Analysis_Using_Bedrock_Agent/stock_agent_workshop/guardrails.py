import pandas as pd
import streamlit as st

def add_result(results, category, details):
    status = "⚠️ Filtered" if details else "✅ OK"
    details_str = ", ".join(details) if details else "-"

    results.append({
        "Check": category,
        "Result": status,
        "Details": details_str
    })

def display_guardrail_result_table(trace_container, assessment):
    """ Show guardrail results in a table """
    results = []

    # Word Policy check
    if 'wordPolicy' in assessment:
        details = []
        for word in assessment['wordPolicy'].get('customWords', []):
            if word.get('action') == 'BLOCKED':
                details.append(f"Found '{word['match']}'")
        add_result(results, "Word check", details)

    # Sensitive Information check
    if 'sensitiveInformationPolicy' in assessment:
        details = []
        for regex in assessment['sensitiveInformationPolicy'].get('regexes', []):
            if regex.get('action') == 'BLOCKED':
                details.append(f"Found '{regex['name']}' pattern")

        for pii in assessment['sensitiveInformationPolicy'].get('piiEntities', []):
            if pii.get('action') in ['BLOCKED', 'ANONYMIZED']:
                details.append(f"Found {pii['type']}")

        add_result(results, "Sensitive info check", details)

    # Content check
    if 'contentPolicy' in assessment:
        details = []
        for filter in assessment['contentPolicy'].get('filters', []):
            if filter.get('action') == 'BLOCKED':
                details.append(f"Found {filter['type']} content")

        add_result(results, "Content check", details)

    # Topic check
    if 'topicPolicy' in assessment:
        details = []
        for topic in assessment['topicPolicy'].get('topics', []):
            if topic.get('action') == 'BLOCKED':
                details.append(f"Found {topic['name']} topic")

        add_result(results, "Topic check", details)

    # Context check
    if 'contextualGroundingPolicy' in assessment:
        details = []
        for filter in assessment['contextualGroundingPolicy'].get('filters', []):
            if filter.get('action') == 'BLOCKED':
                details.append(f"Failed {filter['type']} check")

        add_result(results, "Context check", details)

    if results:
        df = pd.DataFrame(results)
        trace_container.dataframe(df, hide_index=True, use_container_width=True)

def display_guardrail_trace(trace_container, guardrail_trace):
    """ Show guardrail trace info """
    action = guardrail_trace.get('action')

    if 'inputAssessments' in guardrail_trace:
        with trace_container.chat_message("ai"):
            st.markdown("Bedrock Guradrails: Checking input...")

        if action == "NONE":
            trace_container.success("✅ Input OK")

        if action == "INTERVENED":
            trace_container.warning("⚠️ Input filtered")
            for assessment in guardrail_trace.get('inputAssessments', []):
                display_guardrail_result_table(trace_container, assessment)

    else:
        # 'outputAssessments' in guardrail_trace:
        with trace_container.chat_message("ai"):
            st.markdown("Bedrock Guradrails: Check output...")

        if action == "NONE":
            trace_container.success("✅ Output OK")

        if action == "INTERVENED":
            trace_container.warning("⚠️ Output filtered")
            for assessment in guardrail_trace.get('outputAssessments', []):
                display_guardrail_result_table(trace_container, assessment)
