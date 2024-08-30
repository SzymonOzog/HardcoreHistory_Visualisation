import torch
from diffusers import FluxPipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from transformers import pipeline     



pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload() 

prompt = "Diverse group of people gathered around ancient texts and artifacts depicting various mythological scenes"

enhancer = pipeline("summarization", model="gokaygokay/Lamini-Prompt-Enchance", device=0)
prefix = "Enhance the description: "
res = enhancer(prefix + prompt)
print(res[0]['summary_text'])
prompt = res[0]['summary_text']

sz = 768 
image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.Generator("cpu").manual_seed(42),
    height=sz,
    width=sz
).images[0]
image.save("output.png")

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

generator = torch.manual_seed(42)
frames = pipe(image,
              decode_chunk_size=8,
              height=sz,
              width=sz, 
              generator=generator).frames[0]

export_to_video(frames, "enchanced.mp4", fps=7)

# import torch
# from diffusers import CogVideoXPipeline
# from diffusers.utils import export_to_video
#
# pipe = CogVideoXPipeline.from_pretrained(
#     "THUDM/CogVideoX-5b",
#     torch_dtype=torch.bfloat16
# )
#
# pipe.enable_model_cpu_offload()
# pipe.vae.enable_tiling()
#
#
# gs = 6
# video = pipe(
#     prompt=prompt,
#     image=image,
#     num_videos_per_prompt=1,
#     num_inference_steps=50,
#     num_frames=25,
#     guidance_scale=gs,
#     generator=torch.Generator(device="cuda").manual_seed(42),
#     height=sz,
#     width=sz
# ).frames[0]
#
# export_to_video(video, f"output_{gs}.mp4", fps=8)
