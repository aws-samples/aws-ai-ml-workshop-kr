class Dialog {
    static close(active_contexts, session_attributes, intent, messages) {
        intent['state'] = 'Fulfilled'
        return {
            'sessionState': {
                'activeContexts': active_contexts,
                'sessionAttributes': session_attributes,
                'dialogAction': {
                    'type': 'Close'
                },
                'intent': intent
            },
            'requestAttributes': {},
            'messages': messages
        }
    }
    
    static get_slot(slotname, intent)
    {
    try {
        console.log(JSON.stringify(intent['slots']));
        
        if (intent['slots'] && intent['slots'][slotname])
        {
            console.log('Debug:');
            console.log(JSON.stringify(intent['slots'][slotname]));
            var interpretedValue = intent['slots'][slotname]['value']['interpretedValue'];
            var originalValue = intent['slots'][slotname]['value']['originalValue'];
            return interpretedValue;

            // if kwargs.get('preference') == 'interpretedValue':
            //     return interpretedValue
            // elif kwargs.get('preference') == 'originalValue':
            //     return originalValue
            // # where there is no preference
            // elif interpretedValue:
            //     return interpretedValue
            // else
            //     return originalValue
            // }

        }   else return null;

        } catch (error) {
            return null;
        }
        
    }
    
    static set_slot(slotname, slotvalue, intent)
    {
        if (!slotvalue)
            intent['slots'][slotname] = null
        else
        {
            intent['slots'][slotname] = {
                    "value": {
                    "interpretedValue": slotvalue,
                    "originalValue": slotvalue,
                    "resolvedValues": [
                        slotvalue
                    ]
                }
            }
        }
    }
        
        
    static elicit_slot(slotToElicit, active_contexts, session_attributes, intent, messages)
    {
        intent['state'] = 'InProgress';
        
        if (!session_attributes)
            session_attributes = {};
        session_attributes['previous_message'] = JSON.stringify(messages);
        session_attributes['previous_dialog_action_type'] = 'ElicitSlot';
        session_attributes['previous_slot_to_elicit'] = slotToElicit;
        
        return {
            'sessionState': {
                'sessionAttributes': session_attributes,
                'activeContexts': active_contexts,
                'dialogAction': {
                    'type': 'ElicitSlot',
                    'slotToElicit': slotToElicit
                },
                'intent': intent
            },
            'requestAttributes': {},
            'messages': messages
        }
    }

    
    static delegate(active_contexts, session_attributes, intent){
    return {
        'sessionState': {
            'activeContexts': active_contexts,
            'sessionAttributes': session_attributes,
            'dialogAction': {
                'type': 'Delegate'
            },
            'intent': intent,
            'state': 'ReadyForFulfillment'
        },
        'requestAttributes': {}
    }
}


    static get_intent(intent_request) {
        const interpretations = intent_request['interpretations'];
        if (interpretations.length > 0)
            return interpretations[0]['intent']
        else
            return null;
    }


    static get_active_contexts(intent_request) {
        try {
            return intent_request['sessionState'].get('activeContexts');
        } catch (error) {
            return [];
        }
    }

    // static get_context_attribute(active_contexts, context_name, attribute_name) {
    //     try {
    //         //context = list(filter(lambda x: x.get('name') == context_name, active_contexts));

    //         return context[0]['contextAttributes'].get(attribute_name);
    //     } catch (error) {
    //         return null;
    //     }
    // }

    static get_session_attributes(intent_request) {
        try {
            return intent_request['sessionState']['sessionAttributes'];
        } catch (error) {
            return {};
        }
    }


    static get_session_attribute(intent_request, session_attribute) {
        try {
            return intent_request['sessionState']['sessionAttributes'][session_attribute]
        } catch (error) {
            return null;
        }
    }
    
    static elicit_intent(active_contexts, session_attributes, intent, messages) {
    
    intent['state'] = 'Fulfilled';
    
    if (!session_attributes)
        session_attributes = {};
        
    session_attributes['previous_message'] = JSON.stringify(messages);
    session_attributes['previous_dialog_action_type'] = 'ElicitIntent';
    session_attributes['previous_slot_to_elicit'] = null;
    
    return {
        'sessionState': {
            'sessionAttributes': session_attributes,
            'activeContexts': active_contexts,
            'dialogAction': {
                'type': 'ElicitIntent'
            },
            "state": "Fulfilled"
        },
        'requestAttributes': {},
        'messages': messages
    }

    }
    
    static   confirm_intent(active_contexts, session_attributes, intent, messages, previous_state)
    {
    intent['state'] = null;
    
    if (! session_attributes)
        session_attributes = {}
    session_attributes['previous_message'] = JSON.stringify(messages);
    session_attributes['previous_dialog_action_type'] = 'ConfirmIntent';
    session_attributes['previous_slot_to_elicit'] = null;
    
    if (previous_state) {
        session_attributes['previous_dialog_action_type'] = previous_state.get('previous_dialog_action_type');
        session_attributes['previous_slot_to_elicit'] = previous_state.get('previous_slot_to_elicit');
    }
    return {
            'sessionState': {
                'activeContexts': active_contexts,
                'sessionAttributes': session_attributes,
                'dialogAction': {
                    'type': 'ConfirmIntent'
                },
                'intent': intent
            },
            'requestAttributes': {},
            'messages': messages
        }
        
    }
}

module.exports = Dialog;