from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field
import joblib
import uvicorn
from utils.utils import predict_text

app = FastAPI()

# Load the models
sub_category_model = joblib.load('./prepare_models/latest_models/sub_category_model.pkl')
category_model = joblib.load('./prepare_models/latest_models/category_model.pkl')



class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="The input text to be processed")




@app.get("/")
async def health_chech():
    return JSONResponse(content={"message":"API is wokring fine"},status_code=200)


@app.post("/predict_model_1",
           summary="Predict Category and Subcategory",
            description="""This endpoint takes a piece of text and predicts the category and subcategory based on the trained model.
                            \n
                            Category Accuracy: \n
                            Sub-Category Accuracy: \n
                            replace the string with your text input in given request body.
                        """,
)
async def predict(request: TextRequest):
    input_text=request.text.lower()
    # Predict subcategory
    cleaned_text = " ".join(input_text.splitlines())  # Joins lines into a single line

    criminal_info =  [cleaned_text]

    print(criminal_info)

    # Predict the category
    sub_category_prediction = sub_category_model.predict(criminal_info)
    print(f"Sub-Category {sub_category_prediction}")
    predicted_category = category_model.predict(sub_category_prediction)
    print(f"Category {predicted_category}")

    sub_category_prediction=sub_category_prediction[0]
    predicted_category=predicted_category[0]
    return JSONResponse(content={
        "sub_category": sub_category_prediction,
        "category": predicted_category
    },status_code=200)


@app.post("/predict_model_2",
            summary="Predict Category and Subcategory",
            description="""This endpoint takes a piece of text and predicts the category and subcategory based on the trained model.
                            \n
                            Category Accuracy: \n
                            Sub-Category Accuracy: \n
                            replace the string with your text input in given request body.
                        """)          
async def predict_2(request: TextRequest):
    input_text=request.text.lower()
    # Predict subcategory
    cleaned_text = " ".join(input_text.splitlines())  # Joins lines into a single line

    criminal_info =  [cleaned_text]

    predicted_category, sub_category_prediction=predict_text(criminal_info)
    return JSONResponse(content={
        "sub_category": sub_category_prediction,
        "category": predicted_category
    },status_code=200)



# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
