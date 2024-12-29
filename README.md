
# eGlobin Non-Invasive Hemoglobin Level Detection Application with AI

 

 ## PROJECT AIMS 

 This project aims to develop a non-invasive method for hemoglobin measurement
 using deep learning methods in the field of image processing. The application will
 detect anemia and measure hemoglobin levels by analyzing users’ under-eye images. The   main goals include fast and accurate hemoglobin measurement,
 anemia detection, early diagnosis alerts, a user-friendly mobile application and
 increasing data accuracy. Deep learning methods will be used to effectively analyze
 and understand image data. This innovative approach aims to facilitate anemia
 detection, especially in resource-limited areas, by providing accessible and reliable
 health monitoring.


## DATASET


The dataset used for training the CNN algorithm in this project was meticulously prepared under controlled conditions by licensed biomedical scientists at the Centre for Research in Applied Biology, the University of Energy and Natural Resources, Ghana. Below is a detailed description of the dataset creation process:

#### **Data Collection Process**
- Biomedical scientists received specialized training on capturing conjunctival images of children aged 6-59 months who visited assigned healthcare facilities. 
- Data collection was performed using electronic instruments (Kobo Collect v2021.2.4, Massachusetts, USA) deployed on mobile tablets (Samsung Galaxy Tab 7A, Samsung, Vietnam).  
- The system integrated forms to record the following:
  - Biodata of patients, including **hemoglobin levels**, **age**, and **gender**.  
  - Observations and remarks based on the hemoglobin value obtained during laboratory assessments.
  - A high-quality photograph of the conjunctiva (inner lower eyelid) uploaded directly to the database for accessibility.  

#### **Image Capturing Process**
- The lower eyelid was gently pushed back with the thumb, assisted by the index finger, to expose the conjunctiva for photography.  
- All images were captured by laboratory personnel using ambient natural light with a **12MP resolution standard camera** mounted on the tablets.
- Camera spotlights were turned off during photography to avoid excessive shine effects and mitigate the impact of ambient light, which could interfere with model detection and classification accuracy.

#### **Preprocessing**
- To ensure consistency in the dataset, the **triangle thresholding algorithm** and the **entropy grayscale image algorithm** were applied to extract the Region of Interest (ROI) of the conjunctiva.
- These preprocessing steps were implemented to reduce the effect of ambient light on photographs, enhancing the dataset's quality for training purposes.

#### **Source**
- The images were collected from **10 hospitals across Ghana**, ensuring a diverse and representative dataset.  

This high-quality and well-preprocessed dataset played a crucial role in training the CNN model, enabling accurate predictions and robust performance in the application.

#### **Dataset Link**
You can access the dataset at the following link:  
[https://data.mendeley.com/datasets/m53vz6b7fx/1](https://data.mendeley.com/datasets/m53vz6b7fx/1)



## API USAGE




#### Upload an Image
```http
  POST /upload/
```

| Parametre | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `file` | `image file` | **Required**. The image to upload (PNG, JPG, JPEG). |

Response:

| Field | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `segmentedImagePATH` | `str` | Path to the segmented image. |
| `croppedImagePATH` | `str` | Path to the cropped image (Region of Interest). |




#### Predict Hemoglobin Level
```http
  GET /predict
```

| Parametre | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `image_name` | `string` | **Required. The name of the cropped image in the directory.. |
| `age` | `int` | **Required**. The age of the person. |
| `gender` | `string` | **Required**. The gender of the person (male or female). |

Response:

| Field | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `predicted_hblevel` | `float` | The predicted hemoglobin level. |






## Running the Project

Follow the steps below to run the project:

### 1. Starting the Server with Uvicorn

To start the project, open your terminal and navigate to the directory where your project is located. Then, run the following command to start your FastAPI application:

```bash
uvicorn main:app
```


### Running the Mobile Application

After starting the server, follow these steps to run the Flutter mobile application on an emulator:

Make sure the server is running by executing uvicorn main:app in your terminal.

Open a new terminal window and navigate to the Flutter project directory.

If you haven’t installed the required dependencies yet, run:

```bash
flutter pub get
```
Launch an emulator using Android Studio or Xcode, depending on your development environment.

Once the emulator is running, execute the following command to start the mobile application:

```bash
flutter run
```

The Flutter mobile app will be built and launched on the emulator, ready to interact with the FastAPI server.


## DEMO