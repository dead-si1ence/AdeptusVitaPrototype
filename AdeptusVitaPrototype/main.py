import streamlit as st
import joblib
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore


def loadModel(modelPath: str) -> object:
    model = joblib.load(modelPath)
    return model


def loadAndPreprocessImage(image_path, image_size=(224, 224)):
    """Load and preprocess an image from the given path.

    Args:
        image_path (str): The path to the image.
        image_size (tuple, int): The to reshape the images accourding to. Defaults to (224, 224).

    Returns:
        np.array: The preprocessed image.
    """
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img_array = img_to_array(img)  # Convert to array
    img_array = img.resize(
        image_size
    )  # Resize image------------------------------------------------------------------img = img.resize(image_size)
    img_array = np.stack((img_array,) * 3, axis=-1)  # Stack to make 3 channels
    img_array = img_array.reshape(
        (
            image_size[0],
            image_size[1],
            3,
        )
    )  # Ensure the shape is correct
    return preprocess_input(img_array)  # Preprocess the image


def PredictNewInstance(image_path, model, *, showImage=True, imageSize=(224, 224)):
    """
    Preprocess a single image by converting to grayscale, resizing, and extracting features using a pre-trained CNN (VGG16).

    Parameters:
    image_path (str): Path to the image file.
    image_size (tuple): Desired image size. Default is (224, 224).

    Returns:
    X (numpy.ndarray): Array of feature vectors.
    """

    # Load and display image
    # from tensorflow.keras.preprocessing.image import load_img  # type: ignore

    # image = load_img(image_path)
    # display(image) if showImage else None

    # Load VGG16 model + higher level layers
    base_model = VGG16(weights="imagenet", include_top=False, pooling="avg")
    extractor = Model(inputs=base_model.input, outputs=base_model.output)

    # Load and preprocess image
    img = loadAndPreprocessImage(image_path, imageSize)
    img = img.reshape((*imageSize, 3))  # Reshape to desired shape
    img = preprocess_input(img)  # Preprocess the image

    # Extract features
    X = extractor.predict(img[np.newaxis, ...])
    prediction = model.predict(X)

    return (
        "Mild Demented"
        if prediction[0] == 0
        else "Moderate Demented" if prediction[0] == 1 else "Non Demented"
    )


def main():
    # Add logo
    st.title("Adeptus Vita")
    st.image(
        r"d:\bauAstartes\Adeptus Vita\testing\Adeptus_Vita.png",
        use_column_width=False,
        width=250,
    )

    st.write("This is a simple web app prototype to detect dementia using MRI images.")

    # Load the model
    modelPath: str = r"d:\bauAstartes\Adeptus Vita\Models\optimizedXGBoost.pkl"
    model: object = loadModel(modelPath)

    # Load the image
    imagePath = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if imagePath:
        st.image(imagePath, caption="Uploaded Image.", use_column_width=True)
        prediction = PredictNewInstance(imagePath, model)
        st.write(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
