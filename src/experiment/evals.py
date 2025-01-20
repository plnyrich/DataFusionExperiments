from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


def eval_results(y_true, y_pred, classes, section_name, roundPrecision, dst):
	evalMethod = 'macro'
	acc = accuracy_score(y_true, y_pred)
	pre = precision_score(y_true, y_pred, average=evalMethod, zero_division=0) #precision_score(y_true, y_pred, average='binary', zero_division=0)
	rec = recall_score(y_true, y_pred, average=evalMethod, zero_division=0) #recall_score(y_true, y_pred, average='binary', zero_division=0)
	f1s = f1_score(y_true, y_pred, average=evalMethod, zero_division=0)

	dst.write(f'=== {section_name} ===\n')
	dst.write(f'Accuracy     : {round(100 * acc, roundPrecision)} %\n')
	dst.write(f'Precision    : {round(100 * pre, roundPrecision)} %\n')
	dst.write(f'Recall       : {round(100 * rec, roundPrecision)} %\n')
	dst.write(f'F1-Score     : {round(100 * f1s, roundPrecision)} %\n')
	dst.write('\n')

	pre_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=classes)
	rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=classes)

	for idx, c in enumerate(classes):
		if idx > 0:
			dst.write('\n')
		dst.write(f'Precision #{c} : {round(100 * pre_per_class[idx], roundPrecision)} %\n')
		dst.write(f'Recall    #{c} : {round(100 * rec_per_class[idx], roundPrecision)} %\n')

	if evalMethod == 'binary':
		tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
		dst.write('\n')
		dst.write(f'NEG CLASS CNT: {y_true.to_list().count(0)}\n')
		dst.write(f'POS CLASS CNT: {y_true.to_list().count(1)}\n')
		dst.write('\n')
		dst.write(f'TP: {tp} ; FP: {fp}\n')
		dst.write(f'FN: {fn} ; TN: {tn}\n')

	dst.write(f'=== {section_name} ===\n')

	return acc, rec, pre, f1s
