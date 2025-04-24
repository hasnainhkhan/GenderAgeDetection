import numpy as np
import cv2
import pickle

# Load all models
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')  # Cascade Classifier
model_svm = pickle.load(open('./model/model_svm.pickle', mode='rb'))  # Machine Learning Model (SVM)
pca_models = pickle.load(open('./model/pca_dict.pickle', mode='rb'))  # PCA Dictionary
model_pca = pca_models['pca']  # PCA Model
mean_face_arr = pca_models['mean_face']  # Mean Face

# Load age detection model (OpenCV DNN-based model for age prediction)
age_net = cv2.dnn.readNetFromCaffe('./model/deploy_age.prototxt', './model/age_net.caffemodel')
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

def faceRecognitionPipeline(filename, path=True):
    if path:
        # Step-01: Read Image
        img = cv2.imread(filename)  # BGR
    else:
        img = filename  # Array
    
    # Step-02: Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Step-03: Detect Faces (using Haar Cascade)
    faces = haar.detectMultiScale(gray, 1.5, 3)
    predictions = []
    
    for x, y, w, h in faces:
        # Extract Region of Interest (ROI) for face
        roi = gray[y:y + h, x:x + w]
        
        # Step-04: Normalize (0-1)
        roi = roi / 255.0
        
        # Step-05: Resize image (100,100)
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_CUBIC)
        
        # Step-06: Flatten the image (1x10000)
        roi_reshape = roi_resize.reshape(1, 10000)
        
        # Step-07: Subtract Mean Face
        roi_mean = roi_reshape - mean_face_arr
        
        # Step-08: Apply PCA to get Eigen Image
        eigen_image = model_pca.transform(roi_mean)
        
        # Step-09: Inverse PCA for visualization (optional)
        eig_img = model_pca.inverse_transform(eigen_image)
        
        # Step-10: Pass to ML model (SVM) and get predictions
        results = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(eigen_image)
        prob_score_max = prob_score.max()
        
        # Step-11: Age Detection
        blob = cv2.dnn.blobFromImage(roi_resize, 1, (227, 227), (78.4, 87.9, 114.0), swapRB=False)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_LIST[age_preds[0].argmax()]
        
        # Step-12: Generate Report
        text = "%s : %d%%, Age: %s" % (results[0], prob_score_max * 100, age)
        
        # Define color based on results
        if results[0] == 'male':
            color = (255, 255, 0)  # Blue for male
        else:
            color = (255, 0, 255)  # Magenta for female
        
        # Draw Rectangle and Display Text
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), color, -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 5)
        
        # Collect results
        output = {
            'roi': roi,
            'eig_img': eig_img,
            'prediction_name': results[0],
            'score': prob_score_max,
            'age': age
        }
        
        predictions.append(output)

    return img, predictions
