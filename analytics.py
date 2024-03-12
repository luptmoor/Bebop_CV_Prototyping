import pandas as pd
import numpy as np
import os


df = pd.read_csv('Classifications.csv')


folder = 'Labelled'

ground_truth_list = []

for imagename in os.listdir(folder):
    ground_truth = imagename[0]
    imcode = imagename[-12:-4]
    ground_truth_list.append(ground_truth)


df['Ground Truth'] = pd.Series(ground_truth_list)
df[df.columns[1:]].to_csv('Classification_Labelled.csv')


analytics_frame = pd.DataFrame()
analytics_frame['Img Code'] = df['Img code']

for rf in np.linspace(0, 1, 21):


    decisions = df[f"{rf}"]

    analytics_frame[f'Detections {rf}'] = (decisions != 's')



labels = df['Ground Truth']
analytics_frame[f'Obstacles'] = (labels != 's')

analytics_frame.to_csv('Analytics.csv')







