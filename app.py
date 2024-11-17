from fastapi import FastAPI,Form,UploadFile, File
from typing import Optional

from fastapi.responses import JSONResponse
import joblib
import uvicorn
from utils.utils import predict_text
import os
app = FastAPI()


@app.get("/")
async def health_chech():
    return JSONResponse(content={"message":"API is wokring fine"},status_code=200)


@app.post("/second_approach/naive_bayes",
           summary="Predict Category and Subcategory",
            description="""This endpoint takes a piece of text and predicts the category and subcategory based on the model.
                            \n
                            Category Accuracy: 70% \n
                            Sub-Category Accuracy: 40% \n
                            replace the string with your text input in given request body.
                        """)
async def predict(text: str = Form(...),
                  media_file: Optional[UploadFile] = File(None) ):
    input_text=text.lower()

    # Ensure the directory exists
    upload_dir = "./uploaded_files/"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)  
    # Initialize media info
    media_info = None

    if media_file:
        # Save media file
        file_location = os.path.join(upload_dir, media_file.filename)
        with open(file_location, "wb") as file:
            file_content = await media_file.read()
            file.write(file_content)
        # Process media file (optional)
        media_info = {
            "filename": media_file.filename,
            "content_type": media_file.content_type,
            "saved_path": file_location
        }
   

    criminal_info =  [input_text]

    # Load the models
    sub_category_model = joblib.load('./utils/latest_models/second_approach/sub_category_model.pkl')
    category_model = joblib.load('./utils/latest_models/second_approach/category_model.pkl')
    sub_category_prediction = sub_category_model.predict(criminal_info)
    predicted_category = category_model.predict(sub_category_prediction)

    sub_category_prediction=sub_category_prediction[0]
    predicted_category=predicted_category[0]
    
    return JSONResponse(content={
        "sub_category": sub_category_prediction,
        "category": predicted_category,
        "media_info":media_info
    },status_code=200)


@app.post("/third_approach/lstm",
            summary="Predict Category and Subcategory",
            description="""This endpoint takes a piece of text and predicts the category and subcategory based on the trained model.
                            \n
                            Category Accuracy: 82.04% \n
                            Sub-Category Accuracy: 82.04% \n
                            replace the string with your text input in given request body.
                        """)          
async def cyber_crime_classify(text: str = Form(...),media_file: Optional[UploadFile] = File(None)):
    input_text=text.lower()

     # Ensure the directory exists
    upload_dir = "./uploaded_files/"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # Initialize media info
    media_info = None

    if media_file:
        # Save media file
        file_location = os.path.join(upload_dir, media_file.filename)
        with open(file_location, "wb") as file:
            file_content = await media_file.read()
            file.write(file_content)
        # Process media file (optional)
        media_info = {
            "filename": media_file.filename,
            "content_type": media_file.content_type,
            "saved_path": file_location
        }
   

    criminal_info =  [input_text]
   

    
    predicted_category, sub_category_prediction=predict_text(criminal_info)

    
    return JSONResponse(content={
        "sub_category": sub_category_prediction,
        "category": predicted_category,
        "media_info":media_info
    },status_code=200)



# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
