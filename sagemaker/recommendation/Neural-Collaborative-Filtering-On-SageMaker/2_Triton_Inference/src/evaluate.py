import numpy as np
import torch


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    HR, NDCG = [], []

    count = 0
    for user, item, label in test_loader:
#         user = user.to(device)
#         item = item.to(device)
        user = user.cuda()
        item = item.cuda()

        

        predictions = model(user, item)
        # 차원인 1인 것 제거
        predictions = predictions.squeeze()
        
        
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(
				item, indices).cpu().numpy().tolist()

        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))
        
        count += 1
        
#          print(count)        
#         if count == 5:
#             break

#         print("HR: ", HR)
#         print('NDCG: ', NDCG)
        
    return np.mean(HR), np.mean(NDCG)


def predict_metric(model, test_loader, top_k):
    HR, NDCG = [], []

    for i, el in enumerate(test_loader):        
        user = el[0]
        item = el[1]
        
        user_np = user.detach().cpu().numpy()
        item_np = item.detach().cpu().numpy()        
        
        print("user shape: ", user_np.shape)
        #print("user: ", user_np)        
        print("item: ", sorted(item_np))        
        
        user = user.cuda()
        item = item.cuda()

        predictions = model(user, item)
        predictions_numpy = predictions.detach().cpu().numpy()
        print("predictions: ", predictions_numpy.shape)
        #print("predictions: \n", predictions_numpy)        
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(
				item, indices).cpu().numpy().tolist()

        print("recommends: ", recommends)
        gt_item = item[0].item()
        print("gt_item: \n", gt_item)
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))
        
        print("HR: \n", HR)
        print("NDCG: \n", NDCG)        
        print("\n\n")

        if i == 3:
            break

    return recommends


def predict(model, payload, top_k):

    user_np = np.asarray(payload['user'])
    item_np = np.asarray(payload['item'])   
    
    user = torch.from_numpy(user_np)
    item = torch.from_numpy(item_np)
    
    user = user.cuda()
    item = item.cuda()
    
    predictions = model(user, item)
    predictions_numpy = predictions.detach().cpu().numpy()
    # print("predictions: ", predictions_numpy.shape)

    _, indices = torch.topk(predictions, top_k)
    recommends = torch.take(item, indices).cpu().numpy().tolist()

    return recommends


    
    
