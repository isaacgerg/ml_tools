
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=0.5):
    tmp  = (y_pred >= threshold).astype('float32')
    # N = total number of negative labels
    N = np.sum(1.0 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = np.sum(tmp - tmp * y_true)
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=0.5):
    tmp = (y_pred >= threshold).astype('float32')
    # P = total number of positive labels
    P = np.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = np.sum(tmp * y_true)
    return TP/P

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Compute AUC and ROC
thresholds = np.unique(y_pred)[::-1]
numUnique = thresholds.shape[0]
pfaCurve = np.zeros(numUnique)
ptaCurve = np.zeros(numUnique)
for t in range(numUnique):
    pfaCurve[t] = metrics.binary_PFA(y_true, y_pred, threshold=thresholds[t])
    ptaCurve[t] = metrics.binary_PTA(y_true, y_pred, threshold=thresholds[t])

auc = np.trapz(ptaCurve, pfaCurve)
