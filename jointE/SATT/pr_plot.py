import os
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys

from matplotlib.backends.backend_pdf import PdfPages


ff = plt.figure()

MODEL = 'cnn'

def guolv(recall, precision):
    a = [recall[0]]
    b = [precision[0]]
    print len(recall)
    for i in range(1, len(recall)):
        if a[len(a) - 1] == recall[i]:
            if precision[i] > b[len(b)-1]:
                b[len(b)-1] = precision[i]
        else:
            a.append(recall[i])
            b.append(precision[i])
            
    recall = np.array(a)
    precision = np.array(b)
    xnew = np.linspace(recall.min(),recall.max(), 500) #300 represents number of points to make between T.min and T.max  
    print recall
    print precision
    power_smooth = spline(recall,precision,xnew)  
    return xnew, power_smooth

def PrecisionAtRecall(pAll, rAll, rMark):
    length = len(rAll)
    lo = 0
    hi = length - 1
    mark = length >> 1
    error = rMark - rAll[mark]
    while np.abs(error) > 0.005:
        if error > 0:
            hi = mark - 1
        else:
            lo = mark + 1
        mark = (hi + lo) >> 1
        error = rMark - rAll[mark]
    return pAll[mark], rAll[mark], mark


color = ['red', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']

test_model = ['cnn'+'+sen_att']
test_epoch = ['9']
avg_pres = []
for temp, (model, step) in enumerate(zip(test_model, test_epoch)):
    y_scores = np.load(model+'_all_prob' + '_' + step + '.npy')
    y_true = np.load(model+'_all_label' + '_' + step + '.npy')
    y_scores = np.reshape(y_scores,(-1))
    y_true = np.reshape(y_true,(-1))
    precision,recall,threshold = precision_recall_curve(y_true,y_scores)
    p,r,i = PrecisionAtRecall(precision, recall, 0.1)
    print('precison: {}, recall: {}'.format(p, r))
    p,r,i = PrecisionAtRecall(precision, recall, 0.2)
    print('precison: {}, recall: {}'.format(p, r))
    p,r,i = PrecisionAtRecall(precision, recall, 0.3)
    print('precison: {}, recall: {}'.format(p, r))
    average_precision = average_precision_score(y_true, y_scores)
    avg_pres.append(average_precision)
    recall = recall[::-1]
    precision = precision[::-1]
    plt.plot(recall[:], precision[:], lw=2, color=color[1],label="CNN+ATT")

lines_cnn = open('cnn.txt').readlines()
lines_cnn = [t.strip().split()[:2] for t in lines_cnn]
precision_cnn = np.array([t[0] for t in lines_cnn], dtype=np.float32)
recall_cnn = np.array([t[1] for t in lines_cnn], dtype=np.float32)
plt.plot(recall_cnn, precision_cnn, lw=2, color=color[-1], label="CNN+ATT") 


plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.3, 1.0])
plt.xlim([0.0, 0.4])
plt.title('Precision-Recall Area={0:0.4f}'.format(avg_pres[-1]))
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig('sgd_'+MODEL)
plt.plot(range(10), range(10), "o")
plt.show()
ff.savefig("pr.pdf", bbox_inches='tight')
