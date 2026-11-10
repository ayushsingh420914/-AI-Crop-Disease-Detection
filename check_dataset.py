import os

DATASET_PATH = "data/PlantVillage/plantvillage dataset/color"

classes = os.listdir(DATASET_PATH)

print("Total Classes:", len(classes))

total_images = 0

for disease in classes:
    folder = os.path.join(DATASET_PATH, disease)

    if os.path.isdir(folder):
        images = len([
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        total_images += images
        print(disease, ":", images)

print("\nTotal Images:", total_images)