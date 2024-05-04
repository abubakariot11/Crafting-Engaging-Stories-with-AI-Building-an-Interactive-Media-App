import streamlit as st
from clarifai.client.model import Model
import base64
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

load_dotenv()
import os
from clarifai.client.input import Inputs


clarifai_pat = os.getenv("CLARIFAI_PAT")
openai_api_key = os.getenv("OPEN_AI")


def generate_image(user_description, api_key):
    prompt = f"You are a professional comic artist. Based on the below user's description and content, create a proper story comic: {user_description}"
    inference_params = dict(quality="standard", size="1024x1024")
    model_prediction = Model(
        f"https://clarifai.com/openai/dall-e/models/dall-e-3?api_key={api_key}"
    ).predict_by_bytes(
        prompt.encode(), input_type="text", inference_params=inference_params
    )
    output_base64 = model_prediction.outputs[0].data.image.base64
    with open("generated_image.png", "wb") as f:
        f.write(output_base64)
    return "generated_image.png"



def understand_image(base64_image, api_key):
    prompt = "Analyze the content of this image and write a creative, engaging story that brings the scene to life. Describe the characters, setting, and actions in a way that would captivate a young audience:"
    inference_params = dict(temperature=0.2, image_base64=base64_image, api_key=api_key)
    model_prediction = Model(
        "https://clarifai.com/openai/chat-completion/models/gpt-4-vision"
    ).predict_by_bytes(
        prompt.encode(), input_type="text", inference_params=inference_params
    )
    return model_prediction.outputs[0].data.text.raw

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    


def generate_story_from_image(image_bytes, prompt, api_key):
    inference_params = dict(temperature=0.2, max_tokens=100)
    model_prediction = Model("https://clarifai.com/openai/chat-completion/models/openai-gpt-4-vision").predict(
        inputs=[Inputs.get_multimodal_input(input_id="", image_bytes=image_bytes, raw_text=prompt)],
        inference_params=inference_params
    )
    return model_prediction.outputs[0].data.text.raw


def main():
    st.set_page_config(page_title="Interactive Media Creator", layout="wide")
    st.title("Interactive Media Creator")


    with st.sidebar:
        st.header("Controls")
        st.subheader("Image Generation")

        image_description = st.text_area("Description for Image Generation", height=100)
        generate_content_btn = st.button("Generate Image and Story")


    if generate_content_btn and image_description:
        with st.spinner("Creating Image..."):
            # Generate image and story from description
            image_path = generate_image(image_description, clarifai_pat)
            if image_path:
                with open(image_path, "rb") as image_file:
                    st.title("Comic Image")
                    image_path = "generated_image.png"
                    st.image(Image.open(image_path), caption='Generated Image')
    
    
                    file_bytes = image_file.read()
                    prompt = "Analyze the content of this image and write a creative, engaging story that brings the scene to life. Describe the characters, setting, and actions in a way that would captivate a young audience:"
                    story_text = generate_story_from_image(file_bytes, prompt, openai_api_key)
                    st.spinner("Creating Image...")
                    st.header("Generated Story")
                    st.text(story_text)
                    st.success("Content generated successfully!")
            else:
                st.error("Failed to generate content.")


if __name__ == "__main__":
    main()