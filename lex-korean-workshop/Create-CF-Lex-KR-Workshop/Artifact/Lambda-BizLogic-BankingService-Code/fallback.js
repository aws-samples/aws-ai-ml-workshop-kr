const Dialog = require('utils.js') ;

exports.handler =  function (intent_request) {
    
    var active_contexts = Dialog.get_active_contexts(intent_request)
    var session_attributes = Dialog.get_session_attributes(intent_request)
    var intent = Dialog.get_intent(intent_request)
   
    return Dialog.delegate(active_contexts, session_attributes, intent);
   
};