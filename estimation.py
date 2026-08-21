import time
from ultralytics import YOLO

model = YOLO("yolo26n-pose.engine", task="pose")

image_path = "./coco_images" 

start_time = time.perf_counter()
results = model(image_path)
end_time = time.perf_counter()

total_wall_time = end_time - start_time
num_images = len(results)

print("\n===== Per-image processing time =====")

for i, r in enumerate(results, start=1):
    preprocess = r.speed["preprocess"]
    inference = r.speed["inference"]
    postprocess = r.speed["postprocess"]
    processing_total = preprocess + inference + postprocess

    print(
        f"Image {i:03d}: "
        f"preprocess={preprocess:.3f} ms, "
        f"inference={inference:.3f} ms, "
        f"postprocess={postprocess:.3f} ms, "
        f"processing_total={processing_total:.3f} ms"
    )

avg_preprocess = sum(r.speed["preprocess"] for r in results) / num_images
avg_inference = sum(r.speed["inference"] for r in results) / num_images
avg_postprocess = sum(r.speed["postprocess"] for r in results) / num_images
avg_processing_total = avg_preprocess + avg_inference + avg_postprocess
avg_wall_time_ms = total_wall_time / num_images * 1000

print("\n===== Overall =====")
print(f"Average preprocess time: {avg_preprocess:.3f} ms/image")
print(f"Average inference time: {avg_inference:.3f} ms/image")
print(f"Average postprocess time: {avg_postprocess:.3f} ms/image")
print(f"Average processing total: {avg_processing_total:.3f} ms/image")
print(f"Total wall-clock time: {total_wall_time:.3f} s")
print(f"Average wall-clock time: {avg_wall_time_ms:.3f} ms/image")
