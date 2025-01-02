'use strict';

var AWS = require("aws-sdk");
var docClient = new AWS.DynamoDB.DocumentClient();

var table = "BankingBot";
class BankingService {

    static async check_balance(userId, bankName, bankAccount) {
        var UserId = userId ? userId : "user-1";
        var BankName = bankName;
        var BankAccount = bankAccount;
        var Balance = null;
        var params = {
            TableName: table,
            Key: {
                "PK": UserId,
                "SK": `${BankName}#${BankAccount}`
            }
        };
        console.log(`GetItem - params :${JSON.stringify(params)}`)

        const awsrequest = await docClient.get(params);
        const result = await awsrequest.promise();

        if (result.Item) {
            console.log("GetItem succeeded:", JSON.stringify(result.Item, null, 2));
            Balance = result.Item.Balance;
        }
        return Balance;

    }
    static async list_accounts (userId) {
        var UserId = userId ? userId : "user-1";
        var Accounts = null;

        var params = {
          ExpressionAttributeValues: {
            ':userId' : UserId
          },
          KeyConditionExpression: 'PK = :userId',
          TableName: table
        };

        console.log(`Query - params :${JSON.stringify(params)}`)

        const awsrequest = await docClient.query(params);
        const result = await awsrequest.promise();

        if (result.Items) {
            console.log("Query succeeded :", JSON.stringify(result.Items, null, 2));
            Accounts = result.Items;
        }
        return Accounts;

    }
}

module.exports = BankingService;