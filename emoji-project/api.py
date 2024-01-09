from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import RedirectResponse

app = FastAPI(
    title="Emoji Prediction Classifier",
    summary="An API endpoint to predict what emoji would fit best with a short piece of text. Trained on a dataset of US tweets and their corresponding emojis.",
    description="""
# An API endpoint to access an SVM
# Model usage
The model is trained on a dataset of US tweets, and so the input should roughly resemble a tweet for best results. The input should not be more than 280 characters.

## Limitations
The model's dataset is imbalanced, so it may do better at predicting some emojis over others.

## Source
The model is sourced from https://github.com/Pempoczky/emoji-prediction.
    """,
    version="alpha",
)

#testing purposes
class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@app.get("/", description="Root endpoint that redirects to documentation.")
async def root():
    return RedirectResponse(url='/docs')

#testing purposes
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

#testing purposes
@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}


#this is roughly what the final prediction is going to look like

# @app.post("/predict", description="Text classifier endpoint. Input text into the text field to send "
#                                   "request. Text should not be more than 280 characters. "
#                                   "Returns predicted class.",
#           response_model=DigitPredictions,
#           response_description="Emoji class from 1-20 that corresponds to the text")
#
# async def predict(image: UploadFile):
#     try:
#         tensor_image, raw_image = process_image(image)
#     except PIL.UnidentifiedImageError:
#         raise HTTPException(status_code=415, detail="Invalid image")
#     confs: np.ndarray = mnist_classifier.predict(tensor_image)
#     digit_confs = [DigitConfidence(digit=i, confidence=conf) for i, conf in enumerate(confs)]
#
#     return DigitPredictions(predictions=digit_confs)
