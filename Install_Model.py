from transformers import AutoImageProcessor, AutoModelForImageClassification

model_name = "imfarzanansari/skintelligent-acne"

# Load
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Save
save_dir = "./skintelligent-acne/pretrain_model"
processor.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print("Model and processor saved to:", save_dir)
