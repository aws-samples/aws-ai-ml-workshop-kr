const Dialog = require('utils.js') ;
const BankingService = require('banking_service.js') ;

async function validate_slots(intent_request, intent, active_contexts,session_attributes)
{
    const userId = Dialog.get_session_attribute(intent_request, 'customer_id')?Dialog.get_session_attribute(intent_request, 'customer_id'): 'user-1';
    const savedBankName = Dialog.get_session_attribute(intent_request, 'bankName');
    const savedBankAccount = Dialog.get_session_attribute(intent_request, 'bankAccount');
    const bankNameOrigin = Dialog.get_slot('BankNameOrigin', intent)   
    const bankAccountOrigin = Dialog.get_slot('BankAccountOrigin', intent)
    const amountToTransfer = Dialog.get_slot('AmountToTransfer', intent)
    const bankNameDest = Dialog.get_slot('BankNameDest', intent)   
    const bankAccountDest = Dialog.get_slot('BankAccountDest', intent)
    var messages=null;
    
    console.log ("bankNameOrigin :" + bankNameOrigin);
    console.log ("bankAccountOrigin :" + bankAccountOrigin);
    console.log ("savedBankName :" + savedBankName);
    console.log ("savedBankAccount :" + savedBankAccount);
    
    
    if (savedBankName && savedBankAccount)
    {
        prompt = `계좌이체를 진행합니다. 출금계좌로 ${savedBankName} ${savedBankAccount}를 이용하시겠습니까?`;
        console.log(prompt);
        session_attributes.bankName =null;
        session_attributes.bankAccount =null;
        Dialog.set_slot('BankNameOrigin',savedBankName, intent);
        Dialog.set_slot('BankAccountOrigin',savedBankAccount, intent);
                     
        return Dialog.confirm_intent(
                    active_contexts, session_attributes, intent,
                    [{'contentType': 'SSML','content': prompt}],null );
            
    }
    
    if (bankNameOrigin && bankAccountOrigin)
    {
        console.log ("bankNameOrigin ::" + bankNameOrigin);
        console.log ("bankAccountOrigin ::" + bankAccountOrigin);
        var balance = await BankingService.check_balance (userId, bankNameOrigin,  bankAccountOrigin );
        console.log ("balance ::" + balance);
        if (balance)
        {
            if(amountToTransfer)
            {
                if (balance < amountToTransfer)
                 {
                     Dialog.set_slot('BankNameOrigin',null, intent);
                     Dialog.set_slot('BankAccountOrigin',null, intent);
                     Dialog.set_slot('AmountToTransfer',null, intent);
                     messages = [{'contentType': 'PlainText', 'content': `잔액이 부족합니다. 현재 ${bankNameOrigin} ${bankAccountOrigin} 계좌의 출금가능 금액은 ${balance}원입니다.`},{'contentType': 'PlainText', 'content': "어느 은행의 계좌에서 출금할까요?"}];         
                     balance=null;
                     return Dialog.elicit_slot('BankNameOrigin', active_contexts, session_attributes, intent, messages);
                 }
            }
        }
        else 
        {
            Dialog.set_slot('BankNameOrigin',null, intent)
            Dialog.set_slot('BankAccountOrigin',null, intent)
            messages = [{'contentType': 'PlainText', 'content': "계좌가 존재하지 않습니다."},{'contentType': 'PlainText', 'content': "어느 은행의 계좌에서 출금할까요?"}];         
            return Dialog.elicit_slot('BankNameOrigin', active_contexts, session_attributes, intent, messages);
        }
        // var messages = balance?[{'contentType': 'PlainText', 'content': `${bankName} ${bankAccount} 계좌의 출금 가능금액은 ${balance}원입니다.`}]:[{'contentType': 'PlainText', 'content': `계좌가 존재하지 않습니다.`}]
    }
    
    else 
    {
        if (! bankNameOrigin)
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
            
            messages = [{'contentType': 'PlainText', 'content': `${msg} 어느 은행의 계좌에서 인출을 원하시나요?`}];         
            
            return Dialog.elicit_slot('BankNameOrigin', active_contexts, session_attributes, intent, messages);
        }
        else 
        
            return Dialog.delegate(active_contexts, session_attributes, intent);
    }
    return Dialog.delegate(active_contexts, session_attributes, intent);
    
}

exports.handler =  async function (intent_request) {
    
    var active_contexts = Dialog.get_active_contexts(intent_request)
    var session_attributes = Dialog.get_session_attributes(intent_request)
    var intent = Dialog.get_intent(intent_request)
    
    console.log("Handler : Check Balance");

    if (intent['state'] == 'Fulfilled')
        return Dialog.elicit_intent(
                active_contexts, session_attributes, intent, 
                [{'contentType': 'SSML', 'content': "완료!"}]);
    else if (intent['state'] == 'InProgress')            
        return validate_slots(intent_request, intent, active_contexts, session_attributes)
    else if (intent['state'] == 'ReadyForFulfillment')
    {
        const userId = Dialog.get_session_attribute(intent_request, 'customer_id')?Dialog.get_session_attribute(intent_request, 'customer_id'): 'user-1';
        const bankName = Dialog.get_slot('BankNameOrigin', intent)   
        const bankAccount = Dialog.get_slot('BankAccountOrigin', intent)
        // var balance = await BankingService.check_balance (userId, bankName,  bankAccount );
        // var messages = balance?[{'contentType': 'PlainText', 'content': `${bankName} ${bankAccount} 계좌의 출금 가능금액은 ${balance}원입니다.`}]:[{'contentType': 'PlainText', 'content': `계좌가 존재하지 않습니다.`}]
        var messages=[{'contentType': 'PlainText', 'content': `이체가 성공적으로 완료되었습니다.(DDB Update 미구현)`}];
        return Dialog.close(
                active_contexts, session_attributes, intent, messages
                );
    }
    else 
        return Dialog.delegate(active_contexts, session_attributes, intent);

};