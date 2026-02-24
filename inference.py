import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import time
from architecture import * 


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATH_VIDEO = "video.mp4"
PATH_OUTPUT = "output.mp4"

# 1. Recréer l'architecture
model = MobileNetV2_UNet(n_class=1).to(DEVICE)

# 2. Charger les poids
model.load_state_dict(torch.load('model_epoch_5.pth', map_location=DEVICE))
model.eval() # Mode évaluation

print("Modèle chargé et prêt pour l'inférence.")

def process_video(input_path, output_path, model):
    model.eval()
    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))

    transform = T.Compose([
        T.Resize((384, 640)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Traitement en cours...")

    while cap.isOpened():
        start_time = time.time() # Début du chrono

        ret, frame = cap.read()
        if not ret: break

        # 1. Inférence
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.sigmoid(output).cpu().numpy()[0][0]
            mask = (pred > 0.5).astype(np.uint8)

        # 2. Post-processing & Overlay
        mask_hd = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        blue_carpet = frame.copy()
        blue_carpet[mask_hd == 1] = [255, 100, 0]
        result_frame = cv2.addWeighted(blue_carpet, 0.4, frame, 0.6, 0)

        # 3. Calcul des FPS
        end_time = time.time()
        fps_current = 1 / (end_time - start_time)

        # 4. Affichage du texte FPS sur la frame
        cv2.putText(result_frame, f"FPS: {fps_current:.1f}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(result_frame)

    cap.release()
    out.release()
    print(f"Terminé ! Vidéo sauvegardée sous : {output_path}")

process_video(PATH_VIDEO, PATH_OUTPUT, model)