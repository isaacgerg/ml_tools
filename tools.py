
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

#-----------------------------------------------------------------------------------------------------------------------------------------------------
def roc(y_true, y_pred):        
    val0 = y_pred[y_true == 0]
    val1 = y_pred[y_true == 1]
    allSamples = np.unique(np.append(val0, val1))
    thresholds = np.sort(allSamples)[::-1] # make descending, sort makes ascending            
    numAll = allSamples.shape[0]
    pta = np.zeros(numAll)
    pfa = np.zeros(numAll)
    for k in range(numAll):
        pta[k] = binary_PTA(y_true, y_pred, threshold=thresholds[k])
        pfa[k] = binary_PFA(y_true, y_pred, threshold=thresholds[k])
    return (pfa, pta, thresholds)

#-----------------------------------------------------------------------------------------------------------------------------------------------------
def gpuPctCap(fraction):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=fraction), allow_growth=False)
    sess = tf.Session(config=tf.ConfigProto(gpu_options = gpu_options))
    keras.backend.tensorflow_backend.set_session(sess)
    return

ionice
import psutil
p = psutil.Process()
p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
p.ionice(1)
