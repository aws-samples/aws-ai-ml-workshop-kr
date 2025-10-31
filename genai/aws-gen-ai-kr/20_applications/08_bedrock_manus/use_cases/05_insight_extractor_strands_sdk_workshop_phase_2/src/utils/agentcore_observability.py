import logging
from opentelemetry import baggage, context, trace

# Simple logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

###########################
####   Session info    ####
###########################

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    END = '\033[0m'

def set_session_context(session_id, user_type=None, experiment_id=None):
    
    ctx = baggage.set_baggage("session.id", str(session_id))
    #logging.info(f"Session ID '{session_id}' attached to telemetry context")
    logger.info(f"{Colors.GREEN}Session ID '{session_id}' attached to telemetry context{Colors.END}")
    
    if user_type:
        ctx = baggage.set_baggage("user.type", user_type, context=ctx)
        logger.info(f"{Colors.GREEN}user Type '{user_type}' attached to telemetry context{Colors.END}")
    if experiment_id:
        ctx = baggage.set_baggage("experiment.id", experiment_id, context=ctx)
        #logging.info(f"Experiment ID '{experiment_id}' attached to telemetry context")
        logger.info(f"{Colors.GREEN}Experiment ID '{experiment_id}' attached to telemetry context{Colors.END}")
        
    return context.attach(ctx)


###########################
####   Event Helpers    ####
###########################

def add_span_event(span, event_name: str, attributes: dict = None):
    """
    Add an event to the specified span.

    Args:
        span: The OpenTelemetry span to add the event to
        event_name: Name of the event
        attributes: Dictionary of attributes to attach to the event (str, bool, int, float)
    """
    if span and span.is_recording():
        span.add_event(event_name, attributes or {})
    else:
        logger.warning(f"Invalid or non-recording span for event: {event_name}")


def set_span_attribute(span, key: str, value):
    """
    Set an attribute on the specified span.
    
    Args:
        span: The OpenTelemetry span to set the attribute on
        key: The attribute key
        value: The attribute value
    """
    if span and span.is_recording():
        span.set_attribute(key, value)
    else:
        logger.warning(f"Invalid or non-recording span for attribute: {key}")
