
import cv2;
import numpy as np;
import os
import matplotlib.pyplot as plt
import pandas as pd

folder = "week5/week5_raw"



def filter_image(n_rows, resize_factor, image_name = 'test1/1.jpg', y_low = 50, y_high = 200, u_low = 120, u_high = 130, v_low = 120, v_high = 130):
    im = cv2.imread(image_name)
    im = cv2.resize(im, (int(im.shape[1]/resize_factor), int(im.shape[0]/resize_factor)))
    YUV = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    Filtered = np.zeros([YUV.shape[0], YUV.shape[1], 3], dtype=int)
    for y in range(YUV.shape[0]):
        for x in range(n_rows):
            if(YUV[y,x,0] >= y_low and YUV[y,x,0] <= y_high and \
               YUV[y,x,1] >= u_low and YUV[y,x,1] <= u_high and \
               YUV[y,x,2] >= v_low and YUV[y,x,2] <= v_high):
                Filtered[y,x, 1] = 255
    

    original = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    original = cv2.rotate(original, cv2.ROTATE_90_COUNTERCLOCKWISE)

    Filtered = cv2.rotate(Filtered, cv2.ROTATE_90_COUNTERCLOCKWISE)


    return original, Filtered


def classify(risk_factor, verbose=True, plot=True, resize_factor=4, max_elevation=80):
    n_rows = int(max_elevation // resize_factor)
    max_risk = int(n_rows // (1/risk_factor))  # max risk ~ rf
    decisions = []
    image_codes = []
    confidence = 0

    if verbose:
        print(f"Number of rows: {n_rows}");
        print(f"maximum risk: {max_risk}");
        print()

    for imagename in os.listdir(folder):

        path = folder + "/" + imagename
        print(path)
        image_codes.append(imagename)

        # Works on manual testset with our drones and without the plant
        #fimage = filter_color(image_name=path, y_low = 80, y_high = 150, u_low = 0, u_high = 120, v_low = 0, v_high = 155)

        # accepts brighter images to account for the overlooked ground near the control desks, also ignores darker patches to detect less plant
        original_image, filtered_image = filter_image(n_rows, resize_factor=resize_factor, image_name=path, y_low = 100, y_high = 170, u_low = 0, u_high = 120, v_low = 0, v_high = 150)


        centre_length = filtered_image.shape[1] // 3
        lr_length = filtered_image.shape[1] // 2

    
        left = filtered_image[:, :lr_length :]
        centre = filtered_image[:, centre_length:2*centre_length, :]
        right = filtered_image[:, lr_length:, :]

        
        # Centre third computations
        risk_array = np.zeros(centre_length, dtype=int)

        # # Count how many contiguous obstacle pixels there are FROM THE TOP
        # for x in range(centre_length):
        #     for y in range(n_rows):
        #         if centre[y, x, 1] == 0:
        #             risk_array[x] += 1
        #         else:
        #             break

        # risk = np.max(risk_array)

        # Count how many obstacle pixels there are IN TOTAL
        for x in range(centre_length):
            risk_array[x] = n_rows - np.sum(centre[:, x, 1]/255)

        risk = np.max(risk_array)

        
        # Left computations
        size_left = n_rows * lr_length
        count_left = size_left
        for x in range(lr_length):
            for y in range(n_rows):
                if left[y, x, 1] == 0:
                    count_left -= 1
        
        left_preference = count_left/size_left
        
        # size_centre = n_rows * third
        # count_centre = size_centre
        # for x in range(third):
        #     for y in range(n_rows):
        #         if centre[y, x, 1] == 0:
        #             count_centre -= 1
        
        # print(f"Centre: Out of {size_centre} px, {count_centre} are floor ({round(count_centre/size_centre, 3)}).")
        
        
        # Right half computations
        size_right = n_rows * lr_length
        count_right = size_right
        for x in range(lr_length):
            for y in range(n_rows):
                if right[y, x, 1] == 0:
                    count_right -= 1

        right_preference = count_right/size_right
        

        # Decision making and printing
        if verbose:
            print()
            print(f"Right: Out of {size_right} px, {count_right} are floor ({round(right_preference, 3)})")
            print(f"Left: Out of {size_left} px, {count_left} are floor ({round(left_preference, 3)}).")
            print()


            print(f"Risk array: {risk_array}")
            print(f"Risk: {risk}")
            print()

        
        if risk > max_risk:
            confidence -= 2;
            print('Risk too high!')
            if right_preference > left_preference:
                decisions.append('r')
                print('Suggest right.')
            else:
                decisions.append('l')
                print('Suggest left,')
        else:
            confidence += 1;
            print("Suggest moving ahead.")
            decisions.append('s')
        
        confidence = max(min(confidence, 5), 0);
        print(f"Confidence: {confidence}");

        if (confidence == 0):
            print('state: OBSTACLE FOUND');
        elif (confidence >= 2):
            print('state: SAFE');
        
        print()
        print()

        
        if plot:
            displayed = np.vstack((original_image, filtered_image[-n_rows:, :, :]))
            plt.figure()
            plt.imshow(displayed)
            plt.show(block=False)
            dummy = input('Enter to continue')
            plt.close()

    return image_codes, decisions




##########################


# df = pd.DataFrame()

# for rf in np.linspace(0, 1, 21):

#     print(rf)
#     imcodes, decs = classify(rf, verbose=False, plot=False)

#     print(pd.Series(decs))

#     df['Img code'] = pd.Series([imc[-12:-4] for imc in imcodes])
#     df[f'{rf}'] = pd.Series(decs)
#     df.to_csv('Classifications.csv')



imcodes, decs = classify(0.4, resize_factor=1, verbose=True, plot=True)


