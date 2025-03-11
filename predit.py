import torch
from transformers import ViTFeatureExtractor
from PIL import Image
import numpy as np

def predict(multi_view_image):
    multi_view_image_resized = np.array([Image.fromarray((multi_view_image[:,:,i] * 255).astype(np.uint8)).resize((224, 224)) for i in range(3)])
    multi_view_image_resized = np.stack(multi_view_image_resized, axis=-1)

    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    inputs = feature_extractor(images=[multi_view_image_resized], return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    return predicted_class_idx

predicted_class_idx = predict(multi_view_image)
print(f"Predicted class index: {predicted_class_idx}")
