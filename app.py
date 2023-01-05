'''
    A streamlit app with:
        OpenAI GPT-3 task completion
        Huggingface Stable Diffusion Text2Image generation

    A simple dashboard to experiment with GPT-3 and StableDiffusion Models

'''

import os
import time
import openai
from IPython.utils import io
import numpy as np
import streamlit as st
from PIL import Image
from annotated_text import annotated_text

# from files
from utils import simple_clean
from infer import Inference


class cfg:
    # Need your OpenAI API credentials
    username = os.getenv("username")
    password = os.getenv("password")
    hostname = os.getenv("hostname")
    YOUR_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = cfg.YOUR_API_KEY 
 
scheduler_params_default = {
                            "beta_start": 0.0001,
                            "beta_end": 0.02,
                            "beta_schedule": "scaled_linear",
                            "num_train_timesteps": 1000
                           }  

@st.cache(persist=False, allow_output_mutation=True)
def LoadStableDiffusionModel(scheduler="LMSD", scheduler_params=scheduler_params_default):
    infer = Inference(scheduler, scheduler_params)
    return infer

@st.cache(persist=False, allow_output_mutation=True)
def LoadSamplePrompts():
    prompt_list = []
    with open("/mnt/prompts.txt", "r", encoding="UTF-8") as file:
        for line in file:
            line = str(line).strip()
            if len(line) == 0: continue
            prompt_list.append(line)
    file.close()
    return np.array(prompt_list)
    
# APP ICON
st.set_page_config(page_title="Artist AI", layout="wide",)
hide_decoration_bar_style = """ <style> header {visibility: hidden;} </style> """
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
 
# Remove whitespace from the top of the page and sidebar/ css styling
st.markdown(open("static/css.txt").read(), unsafe_allow_html=True)
 
# sample prompts from the file
PROMPTS = LoadSamplePrompts()
 
# SIDEBAR
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 250px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 250px;
        margin-left: -250px;
    }
    </style>
    """, unsafe_allow_html=True,
)
 
with st.sidebar:
    for i in range(10):
        st.write("")
    side_top = '<b style="font-family:Sans-Serif; color:Blue; font-size: 18px;">ML Task</b>'
    st.markdown(side_top, unsafe_allow_html=True)
    
    task = st.selectbox("Select task", ("NLP Task", "Text2Image"))
    
    if task == "NLP Task":
        st.write(" **Uses OpenAI GPT-3 API** ")
    if task == "Text2Image":
        st.write(" **Uses Stable Diffusion Model** ")
    
    for _ in range(25):
        st.write("")
    st.markdown(":copyright: rnepal2", unsafe_allow_html=True)

    
# NAVIGATION BAR
st.markdown(open("static/navbar.txt").read(), unsafe_allow_html=True) 
 
# columns
_, left, _, right, _ = st.columns((0.01, 0.35, 0.050, 0.40, 0.05))
 
# Initialization
submit_sentiment = False
submit_t2i = False
submit_task_comp = False
    
with left:
    if task == "NLP Task":
        st.image("static/openai_logo.png", width=150)
        
        tab1, tab2 = st.tabs(["Task Completion", "Sentiment Classification"])
        
        with tab1:
            st.write("""
                      **A general task completion AI capable for varieties of NLP tasks. It can complete a general task based on description or
                        a description combined with a context. Example tasks: Q/A, Keywords extraction, Language translation, Text generation, 
                        Code generation, etc. and many more.**
                     """
                     )

            task_type = st.radio(label="Task Type", options=("General Task", "Contextual Task"), index=0, horizontal=True)
            
            if task_type == "General Task":
                task_description = st.text_area(label="Describe the task", value="", height=20)
                input_text = ""
            else:
                task_description = st.text_area(label="Describe the task", value="", height=20)
                input_text = st.text_area(label="Context text", value="", height=100)

            submit_task_comp = st.button(label="Submit", key="submit_tc")

            if task_type == "Contextual Task":
                if len(input_text.split()) > 0:
                    _prompt = simple_clean(input_text)
                    prompt = f"{task_description}:\n\n {_prompt}"
                else:
                    if submit_task_comp:
                        st.error("Enter a valid context text.")
                        st.stop()
            else:
                prompt = f"{task_description.strip()}"

        with tab2:
            st.write(" **NLP with GPT-3 using OpenAI API. Classify the sentiment of a text - binary or overall sentiment type.** ")

            task_type = st.radio(label="Sentiment", options=("Binary", "Type"), index=0, horizontal=True)
            token_length = st.slider(label="Text (Token) Size", min_value=32, max_value=1024, value=64, step=32)
            input_text = st.text_area(label="Text", value="", height=100)
            submit_sentiment = st.button(label="Submit", key="submit_sentiment")

            if submit_sentiment:
                if len(input_text.split()) < 1:
                        st.error("Enter a valid text.")
                        st.stop()
                else:
                    if task_type == "Binary":
                        direction = "Classify the sentiment of the text below: "
                        prompt = f"{direction}:\n\n {input_text}"
                    if task_type == "Type":
                        direction = "Find the sentiment tone of the text below: "
                        prompt = f"{direction}:\n\n {input_text}"
    
    if task == "Text2Image":
        st.image("static/stabilityai_logo.png", width=150)
        tab1, tab2 = st.tabs(["Text2Image", "Image2Image"])
        
        in_tab_2 = False

        with tab1:
            scheduler = st.radio(label="Scheduler", options=("LMSD", "DDIM", "PNDM"), index=0, horizontal=True)

            scheduler_params = None
            with st.expander("Set scheduler parameters"):
                beta_start = st.slider(label="beta_start", min_value=0.0, max_value=0.001, value=0.0001, step=0.00005)
                beta_end = st.slider(label="beta_end", min_value=0.001, max_value=0.1, value=0.02, step=0.002)
                beta_schedule = st.radio(label="beta_schedule", options=("scaled_linear", "linear"), index=0, horizontal=True)
                num_train_steps = st.slider(label="num_train_steps", min_value=100, max_value=5000, value=1000, step=100)

                scheduler_params  = {
                                        "beta_start": beta_start,
                                        "beta_end": beta_end,
                                        "beta_schedule": beta_schedule,
                                        "num_train_timesteps": num_train_steps
                               }
            if not scheduler_params:
                scheduler_params=scheduler_params_default
            infer_obj = LoadStableDiffusionModel(scheduler=scheduler, scheduler_params=scheduler_params)
            
            default_prompt = "Anthropomorphic blue-gray furry wolf, digital painting, epic, extremely detailed protrait view, intricate, oil painting, Greg Rutkowski."
            auto_prompt = st.button(label="Give me a sample", key="sample_prompt")
            sample_prompt = np.random.choice(PROMPTS, size=1)[0]
            if auto_prompt:
                sample_prompt = f"[Copy & Paste] {sample_prompt}"
                prompt_text = st.info(body=sample_prompt, icon="ℹ️")
            prompt_text = st.text_area(label="Prompt Text", value=default_prompt, key="enter_prompt")
                            
            # Params selection
            side_top = '<b style="font-family:Sans-Serif; color:Blue; font-size: 18px;">Model Parameters</b>'
            st.markdown(side_top, unsafe_allow_html=True)
            
            with st.expander("Set model parameters"):
                eta = st.slider(label="eta", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
                scale = st.slider(label="guidance_scale", min_value=5.0, max_value=25.0, value=7.5, step=0.5)
                steps = st.slider(label="num_inference_steps", min_value=25, max_value=250, value=75, step=25)
                seed = st.number_input(label="seed", min_value=0, max_value=2**32-1, value=12345, step=None)

                model_params =     {
                                        "eta": eta,
                                        "guidance_scale": scale,
                                        "height": 512,
                                        "width": 512,
                                        "num_inference_steps": steps,
                                        "output_type": "pil",   
                                    }
            
            print_params = model_params.copy()
            del print_params["output_type"]
            print_params["seed"] = seed
            st.write("")
            submit_t2i = st.button(label="Generate Image", key="submit_txt2img")
            
        with tab2:
            in_tab_2 = True
            st.info("Not implemented yet!")
 
 
with right:
    if task == "NLP Task":
        for _ in range(8):
            st.write("")
            
        if submit_task_comp:
            if len(prompt.split()) < 1:
                st.error("Enter valid input text and task!")
                st.stop() 

            with st.spinner("Running..."):
                response = openai.Completion.create(
                            model="text-davinci-002",
                            prompt=prompt,
                            temperature=0,
                            max_tokens=1024,
                            top_p=1.0,
                            frequency_penalty=0.0,
                            presence_penalty=0.0
                        )
            ans = response.choices[0].text
            text = '<b style="font-family:Sans-Serif; color:Blue; font-size: 20px;">Answer</b>'
            st.markdown(text, unsafe_allow_html=True)
            st.write(f"{ans}")

        if submit_sentiment:
            if len(prompt.split()) < 5:
                st.error("Enter valid input text and task!")
                st.stop() 

            if task_type == "Binary":
                with st.spinner("Running..."):
                    response = openai.Completion.create(
                                                        model="text-davinci-002",
                                                        prompt=prompt,
                                                        temperature=0,
                                                        max_tokens=token_length,
                                                        top_p=1.0,
                                                        frequency_penalty=0.0,
                                                        presence_penalty=0.0
                                                    )
            if task_type == "Type":
                with st.spinner("Running..."):
                    response = openai.Completion.create(
                                                        model="text-davinci-001",
                                                        prompt=prompt,
                                                        temperature=0,
                                                        max_tokens=token_length,
                                                        top_p=1.0,
                                                        frequency_penalty=0.0,
                                                        presence_penalty=0.0
                                                    )
            ans = response.choices[0].text

            text = '<b style="font-family:Sans-Serif; color:Blue; font-size: 20px;">Answer</b>'
            st.markdown(text, unsafe_allow_html=True)
            if "positive" in ans.lower():
                rest = ans.replace("positive", "").replace(".", "")
                post = annotated_text(rest, ("positive", "green"))
            elif "negative" in ans.lower():
                rest = ans.replace("negative", "").replace(".", "")
                post = annotated_text(rest, ("negative", "red"))
            else:
                post = ans
            st.write(f"{post}")
            
    if submit_t2i and task == "Text2Image":
        for _ in range(5):
            st.write("")
            
        start = time.time()
        with st.spinner("Running Image Generator..."):
            try:
                image = infer_obj(prompt=prompt_text, seed=seed, params=model_params,)
            except Exception as e:
                if "memory" in str(e).lower():
                    st.error("Wating for GPU memory quota. Please wait and submit your request again.")
                    st.stop()
                else:
                    st.error("Error raised! Please run again with different inputs.")
                    st.stop()
                    
        lapse = round(time.time() - start)
        st.write(" **AI Generated Image** ")
        st.image(image, caption="Output image")
        st.write(f"Run time: {lapse}sec.")

        num_list = []
        prev_images = os.listdir("/mnt/data/")
        prev_images = [img for img in prev_images if ".png" in img]
        if len(prev_images) > 0:
            for img in prev_images:
                try:
                    v = int(img.split(".")[-2].split("_")[-1])
                    num_list.append(v)
                except: pass
        else:
            num_list.append(0)
        n = min(max(num_list), 99)
        
        image_name = f"diffusion_image_{n+1}.png"
        img_path = f"/mnt/data/diffusion_image_{n+1}.png"
        image.save(img_path)

        with open(img_path, "rb") as file:
             btn = st.download_button(
                     label="Download",
                     data=file,
                     file_name=image_name,
                     mime="file/png"
                   )
    
    # Default image loading
    if not submit_t2i and task == "Text2Image":
        for _ in range(5):
            st.write("")
        img_path = "/mnt/data/furry_wolf_1.png"
        if os.path.exists(img_path):
            if prompt_text == default_prompt:
                st.write(" **Generated Image** ")
                show_img = Image.open(img_path)
                st.image(show_img, caption="Furry Wolf")