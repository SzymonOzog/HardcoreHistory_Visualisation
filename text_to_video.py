import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from transformers import pipeline     

prompt = "Diverse group of people gathered around ancient texts and artifacts depicting various mythological scenes"

enhancer = pipeline("summarization", model="gokaygokay/Lamini-Prompt-Enchance", device=0)
prefix = "Enhance the description: "
res = enhancer(prefix + prompt)
print(res[0]['summary_text'])
prompt = res[0]['summary_text']

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

gs = 6
width = 1024
height = 576
seed = 410

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=25,
    guidance_scale=gs,
    generator=torch.Generator(device="cuda").manual_seed(seed),
    # height=height,
    # width=width
).frames[0]

export_to_video(video, f"output_{gs}.mp4", fps=8)
