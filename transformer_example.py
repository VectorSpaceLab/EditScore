import dotenv

dotenv.load_dotenv(override=True)

from PIL import Image
from editscore import EditScore

model_path = "/share/project/jiahao/LLaMA-Factory2/output/merge_v7-2_8models_omnigen2-4samples_gpt4-1_range_0to25"

scorer = EditScore(
    backbone="qwen25vl",
    model_name_or_path=model_path,
    score_range=25,
    num_pass=1, # the number of passes for the model to evaluate the image
)

input_image = Image.open("example_images/input.png")
output_image = Image.open("example_images/output.png")
instruction = "Adjust the background to a glass wall."

result = scorer.evaluate([input_image, output_image], instruction)
print(result)