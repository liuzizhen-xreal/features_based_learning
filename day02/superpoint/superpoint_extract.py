from lightglue import SuperPoint
from lightglue.utils import load_image
from pathlib import Path

base_dir = Path(__file__).resolve().parent
image_path = base_dir.parent.parent / "dataset" / "images.jpg"

extractor = SuperPoint(max_num_keypoints=1000).eval().to("cuda")
image = load_image("dataset/images.jpg").to("cuda")

features = extractor.extract(image, resize=None)

for name in ["keypoints", "keypoint_scores", "descriptors"]:
    value = features[name]
    print(name, value.shape, value.dtype, value.device)