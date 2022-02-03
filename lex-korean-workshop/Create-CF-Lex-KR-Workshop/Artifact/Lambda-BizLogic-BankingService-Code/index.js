const transfer = require('transfer.js');
const check_balance = require('check_balance.js');
const fallback = require('fallback.js');
const Dialog = require('utils.js') ;

async function dispatch(intentRequest, callback) {

    var next_state=null;
    
    console.log(intentRequest);
    var intent = Dialog.get_intent(intentRequest);
    var intent_name = intent['name'];
    
    // Dispatch to the respective bot's intent handlers
    if (intent_name == 'Transfer')
         next_state = 	await transfer.handler(intentRequest);
    else if (intent_name == 'CheckBalance')
         next_state = 	await check_balance.handler(intentRequest);
    else if (intent_name == 'FallbackIntent')
         next_state = 	fallback.handler(intentRequest);

    return callback(next_state);
}
 
// --------------- Main handler -----------------------
 
// Route the incoming request based on intent.
// The JSON body of the request is provided in the event slot.
exports.handler = (event, context, callback) => {
    
    try {
        dispatch(event,
            (response) => {
                console.log("Response : ");
                console.log(response);
                callback(null, response);
            });
    } catch (err) {
        callback(err);
    }
};