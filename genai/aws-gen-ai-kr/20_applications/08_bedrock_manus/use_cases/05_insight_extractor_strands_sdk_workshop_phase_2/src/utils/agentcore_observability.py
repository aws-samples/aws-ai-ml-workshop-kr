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
