const Dialog = require('utils.js') ;
const BankingService = require('banking_service.js') ;


exports.handler =  async function (intent_request) {
    
    var active_contexts = Dialog.get_active_contexts(intent_request)
    var session_attributes = Dialog.get_session_attributes(intent_request)
    var intent = Dialog.get_intent(intent_request)
    const userId = Dialog.get_session_attribute(intent_request, 'customer_id')?Dialog.get_session_attribute(intent_request, 'customer_id'): 'user-1';
    const bankName = Dialog.get_slot('BankName', intent)   
    const bankAccount = Dialog.get_slot('BankAccount', intent)
        
    console.log("Handler : Check Balance");

    if (intent['state'] == 'Fulfilled')
        return Dialog.elicit_intent(
                active_contexts, session_attributes, intent, 
                [{'contentType': 'SSML', 'content': "완료!"}]);
    else if (intent['state'] == 'ReadyForFulfillment')
    {


        var balance = await BankingService.check_balance (userId, bankName,  bankAccount );
        var messages = balance?[{'contentType': 'PlainText', 'content': `${bankName} ${bankAccount} 계좌의 출금 가능금액은 ${balance}원입니다.`}]:[{'contentType': 'PlainText', 'content': `계좌가 존재하지 않습니다.`}]
        var session_attributes={bankName,bankAccount};
        
        return Dialog.close(
                active_contexts, session_attributes, intent, messages
                );
    }
    else 
    {
        if (! bankName)
        {
            var accounts = await BankingService.list_accounts (userId);
            var msg = "";
            
            if (accounts)
            {
                msg +="귀하는 현재 "
                for (var i in accounts)
                {
                    if (i != 0) msg +=", ";
                    msg +=`${accounts[i].SK.replace('#', ' ') } `;
                }
                msg +=" 계좌가 있습니다. "
            }   
            
            messages = [{'contentType': 'PlainText', 'content': `${msg} 어느 은행의 계좌를 확인하시겠습니까?`}];         
            
            return Dialog.elicit_slot('BankName', active_contexts, session_attributes, intent, messages);
        }
        else 
        
            return Dialog.delegate(active_contexts, session_attributes, intent);
    }
};