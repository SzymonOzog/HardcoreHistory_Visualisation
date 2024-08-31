import torch
from diffusers import FluxPipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from transformers import pipeline     
import os

input = [
("Like many of you, I am a fan of ancient mythology.", "Diverse group of people gathered around ancient texts and artifacts depicting various mythological scenes", 5),
("The stories of creation or how human beings came to be,", "Animated sequence showing different creation myths from various cultures, transitioning from chaos to order", 6),
("or tales that involve gods and heroes and monsters,", "Montage of Greek gods on Mount Olympus, heroes battling mythical beasts, and monsters lurking in ancient landscapes", 6),
("and sometimes just regular people who go through interesting sorts of events or travels or whatnot.", "Ordinary people from ancient times encountering extraordinary mythical creatures and magical events", 7),
("And oftentimes these mythological stories are meant to impart lessons.", "Ancient storyteller gesturing to attentive listeners, with thought bubbles appearing above their heads showing moral lessons", 6),
("We're supposed to learn something from them, what to do, what not to do.", "Split screen showing characters making right and wrong choices based on mythological lessons", 6),
("You're tempted to almost say at the end of all of them, and the moral of the story is, right, what are we supposed to learn from this?", "Animated book of myths closing, with a large question mark appearing, transforming into various symbols representing different moral lessons", 8),
("Hunter S. Thompson used to call it in his columns the wisdom.", "Hunter S. Thompson at his typewriter, with the word 'wisdom' appearing in smoke above him, surrounded by mythological imagery", 5),
("And some of my favorite mythological stories are cautionary tales,", "Montage of famous cautionary myths like Pandora's Box and the Midas Touch, with consequences clearly shown", 5),
("examples of what can happen if we're not careful.", "Series of myth-inspired scenes showing careless actions leading to disastrous results in ancient settings", 5),
("And one of my favorite versions of that kind of story, that kind of mythological teaching tool, is the famous story of Daedalus and Icarus.", "Ancient Greek pottery coming to life, depicting the story of Daedalus and Icarus unfolding", 9),
("If you know your ancient greek philosophy, you will recall that Daedalus is a master craftsman, an inventor.", "Daedalus in his ancient Greek workshop, surrounded by intricate inventions and blueprints", 8),
("He can seemingly make anything.", "Daedalus crafting a series of increasingly complex and magical devices in rapid succession", 5),
("He's the one who built the famous labyrinth that held the Minotaur.", "Daedalus overseeing the construction of the massive labyrinth, with the silhouette of the Minotaur visible within", 6),
("And it was the king of Minoa, the cretan area on the island of Crete, that had Daedalus build this for him.", "Map of ancient Crete zooming in to show the king commanding Daedalus to build the labyrinth", 8),
("But at a certain point, he turns against Daedalus and imprisons Daedalus in Icarus.", "The king's guards seizing Daedalus and his son Icarus, locking them in a high tower", 7),
("But of course, when you imprison one of the great inventors of all time, he's going to try to invent a way to get out.", "Daedalus in the prison tower, mind visibly racing as he sketches various escape plans", 8),
("And in this case, he does. He creates wings for he and his son,", "Daedalus gathering feathers and wax, beginning to construct two sets of wings", 6),
("wings made of multiple different materials, including things like feathers and beeswax.", "Close-up of Daedalus' hands meticulously crafting the wings, attaching feathers with wax", 7),
("And he and his son are going to be able to fly out of this prison.", "Daedalus and Icarus on the tower ledge, strapping on their newly made wings", 6),
("But Daedalus warns his son before doing so,", "Daedalus with a serious expression, gesturing cautiously to an excited Icarus", 5),
("he tells him not to get complacent and allow himself to fly too close to the water,", "Split-screen showing Icarus flying safely vs. too low over choppy seas", 7),
("because if you're too low, the moisture, he says, from the sea, will ruin the wings and you'll lose your power of flight and you'll crash.", "Animation of wings absorbing moisture, falling apart, and a figure plummeting towards the sea", 9),
("Conversely, he warns him about getting filled with hubris and forgetting how dangerous this is", "Icarus looking overconfident, imagining himself soaring higher than birds and clouds", 7),
("and allowing himself to fly too high.", "Icarus ascending rapidly, leaving his cautious father behind", 5),
("Because if he does that, the sun will melt the beeswax that hold these wings together, and you'll plummet and fall.", "Sun melting the wings of Icarus as he flies towards it, feathers falling apart and Icarus beginning to fall", 9),
("And of course, being an ancient greek mythological tale, how would it work if everything just went fine? And of course, it doesn't.", "Greek amphitheater with a storyteller dramatically gesturing, audience gasping", 8),
("Icarus forgets his father's warnings, gets taken sort of over by the enthusiasm that happens when a human being gets a chance to fly like a bird,", "Icarus soaring joyfully through clouds, performing aerial acrobatics, clearly forgetting his father's warnings", 9),
("allows himself to fly too high, and the sun melts, the bees wax, the wings fall apart,", "Icarus approaching the sun, his wings starting to melt and disintegrate as he realizes his mistake", 7),
("and Icarus plunges into the sea and dies.", "Icarus falling from the sky, splashing into the sea with Daedalus watching helplessly from a distance, feathers floating on the water's surface", 5)
]

width = 1024
height = 576
fps = 7
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload() 
enhancer = pipeline("summarization", model="gokaygokay/Lamini-Prompt-Enchance", device=0)

for i, (fragment, prompt, time) in enumerate(input):

  dir = f"{i}_{fragment}"
  os.makedirs(dir, exist_ok=True)

  prefix = "Enhance the description: "
  res = enhancer(prefix + prompt)
  print(res[0]['summary_text'])
  res[0]['summary_text']

  for seed in [42, 410, 1337]:
    for gs in [0, 2, 6]:
      for enchance in [True, False]:
        image = pipe(
            res[0]['summary_text'] if enchance else prompt,
            guidance_scale=gs,
            num_inference_steps=50,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(seed),
            height=height,
            width=width
            ).images[0]
        image.save(f"{dir}/output_{seed}_{gs}_{enchance}.png")
