def checkAccuracy (model, test_dataloader):

	model.eval()
	predicted=[]
	actuals=[]
	for i, (inputs, targets) in tqdm.tqdm(
		enumerate(test_dl), total = len(test_dl), leave=False):
		with torch.no_grad():
			logits = model(inputs.float())
			logits = logits.cpu()
			inputs=inputs.cpu()
			targets=targets.cpu()
		preds = torch.argmax(F.softmax(logits, dim=1), axis=1).numpy()
		targets = targets.numpy()
		actuals = np.concatenate((actuals,targets))
		predicted = np.concatenate((predicted,preds))

	acc = (predicted == actuals).mean()
	print(f"\n ODENet accuracy: {acc}")
	print(f"\n  Number of tunable parameters: {count_parameters(odenet)}")


