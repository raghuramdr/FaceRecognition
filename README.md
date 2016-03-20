# FaceRecognition
Face recognition using SIFT
The AT&T database is used. It contains around 400 images of 10 subjects, with 10 images per subject.
The images are captured in various poses. SIFT features are computed for all the images in the training class.
When a query image is provided, SIFT feature descriptors are computed for the query image and matched with all the SIFT descriptors in the database. The image with the maximum number of matches is the matching image. 
