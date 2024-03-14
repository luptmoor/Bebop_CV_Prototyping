import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

analytics_frame = pd.read_csv('Analytics.csv')

TPRs = []
FPRs = []
for rf in np.linspace(0, 1, 21):

    TPs = 0
    FPs = 0
    TNs = 0
    FNs = 0

    for i in range(100):

        # Positives
        if analytics_frame[f'Detections {rf}'].iloc[i] == True:
            if analytics_frame['Obstacles'].iloc[i] == True:
                TPs += 1
            else:
                FPs += 1
        # Negatives
        else:
            if analytics_frame['Obstacles'].iloc[i] == True:
                FNs += 1
            else:
                TNs += 1
    
    TPR = TPs / (TPs + FNs)  # fraction of true positives among all actuallly positives
    FPR = FPs / (FPs + TNs)  # fraction of false positives among all actually negatives


    TPRs.append(TPR)
    FPRs.append(FPR)
    # print(f"For risk factor {rf}, {TPs} true positives, {TNs} true negatives, {FPs} false positives and {FNs} false negatives were detected ({TPs + TNs + FPs + FNs} in total)")
    # dummy = input('break')

resultframe = pd.DataFrame();
resultframe['RF'] = pd.Series(np.linspace(0, 1, 21))
resultframe['TPR'] = pd.Series(TPRs);
resultframe['FPR'] = pd.Series(FPRs);

resultframe.to_csv('Results.csv')


plt.plot(FPRs, TPRs)

xs = np.linspace(0, 1, 10)
plt.plot(xs, xs, 'k--')

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(['Our approach', 'Random guessing'])
plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.grid()
plt.title('Counting non-floor pixels per vertical')
plt.show()