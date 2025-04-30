import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from train.rendering import render_single


def load_img(path):
    transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            # RandomCenterCrop(0.5),
            transforms.Resize(
                224,
                antialias=True,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(224),
            transforms.ConvertImageDtype(torch.float),
        ]
    )
    # img = Image.open(path)
    img = Image.open(path).convert("RGB")  # Ensure the image has 3 channels (RGB)
    img = transform(img)
    # print(f"Loaded image shape: {img.shape}") 
    return img


scene_paths = [
    # [
        # "datasets/dataset_real_5_231024/intellab_01/sensor_0/image_proc/02351.jpg",
        # "datasets/dataset_real_5_231024/intellab_01/sensor_2/image_proc/02479.jpg",
        # "datasets/dataset_real_5_231024/intellab_01/sensor_2/image_proc/02491.jpg",
    # ],
    # [
        # "datasets/dataset_real_5_231024/sn-corridor_01/sensor_0/image_proc/00977.jpg",
        # "datasets/dataset_real_5_231024/sn-corridor_01/sensor_2/image_proc/01412.jpg",
    # ],
    # [
    #     "datasets/dataset_real_5_231024/sn-corridor_01/sensor_2/image_proc/03431.jpg",
    #     "datasets/dataset_real_5_231024/sn-corridor_01/sensor_2/image_proc/03377.jpg",
    #     "datasets/dataset_real_5_231024/sn-corridor_01/sensor_2/image_proc/03407.jpg",
    # ],
    # [
    #     "datasets/dataset_real_5_231024/sn05_01/sensor_1/image_proc/03052.jpg",
    #     "datasets/dataset_real_5_231024/sn05_01/sensor_1/image_proc/00562.jpg",
    #     "datasets/dataset_real_5_231024/sn05_01/sensor_2/image_proc/01038.jpg",
    # ],
    # [
    #     "datasets/dataset_real_5_231024/sn05_01/sensor_2/image_proc/04189.jpg",
    #     "datasets/dataset_real_5_231024/sn05_01/sensor_2/image_proc/00484.jpg",
    #     "datasets/dataset_real_5_231024/sn05_01/sensor_0/image_proc/01057.jpg",
    # ],
    # [
    #     "/workspace/shiyuan_ws/CoViS-Net/mytest_1.jpeg",
    #     "/workspace/shiyuan_ws/CoViS-Net/mytest_2.jpeg",
    #     "/workspace/shiyuan_ws/CoViS-Net/mytest_3.jpeg",
    # ],
    # [
    #     "/workspace/shiyuan_ws/CoViS-Net/mytest2_1.png",
    #     "/workspace/shiyuan_ws/CoViS-Net/mytest2_2.png",
    # ]
    # [
    [
        "/workspace/shiyuan_ws/CoViS-Net/test_img/16-1.png",
        "/workspace/shiyuan_ws/CoViS-Net/test_img/16-2.png",
        # # "/workspace/shiyuan_ws/CoViS-Net/test_img/5-3.jpeg",

    ]

]

CUDA = False
def run(model_base):
    if not CUDA:
        enc = torch.jit.load(f"models/{model_base}_float32_jit_cpu_enc.ts")
        msg = torch.jit.load(f"models/{model_base}_float32_jit_cpu_msg.ts")
        post = torch.jit.load(f"models/{model_base}_float32_jit_cpu_post.ts")
        bev = torch.jit.load(f"models/{model_base}_float32_jit_cpu_bev.ts")
        bev_dec = torch.jit.load(f"models/{model_base}_float32_jit_cpu_bevdec.ts")
    else:
        enc = torch.jit.load(f"models/{model_base}_float32_jit_cuda_enc.ts")
        msg = torch.jit.load(f"models/{model_base}_float32_jit_cuda_msg.ts")
        post = torch.jit.load(f"models/{model_base}_float32_jit_cuda_post.ts")
        bev = torch.jit.load(f"models/{model_base}_float32_jit_cuda_bev.ts")
        bev_dec = torch.jit.load(f"models/{model_base}_float32_jit_cuda_bevdec.ts")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("cuda available:", torch.cuda.is_available())

    for scene_path in scene_paths:
        print("scene_path", scene_path)
        data = {
            "img": torch.stack(
                [load_img(path) for i, path in enumerate(scene_path)], dim=0
            ),
            "encs": None,
            "edge_index": [[], []],
            "edge_preds": {
                "msg": [],
                "pos": [],
                "rot": [],
                "pos_var": [],
                "rot_var": [],
            },
            "node_preds": [],
        }
        with torch.no_grad():
            if not CUDA:
                data["encs"] = [enc(img.unsqueeze(0)) for img in data["img"]]
            else:
                data["encs"] = [enc(img.unsqueeze(0).to(device)) for img in data["img"]]
            
            for i, enc_i in enumerate(data["encs"]):
                for j, enc_j in enumerate(data["encs"]):
                    if i == j:
                        continue
                    data["edge_index"][0].append(j)
                    data["edge_index"][1].append(i)

                    m = msg(enc_i, enc_j)
                    data["edge_preds"]["msg"].append(m)

                    pos, pos_var, heading, heading_var = post(m)
                    # Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw)
                    q = heading.squeeze()  # Assuming heading is a tensor of shape [1, 4]
                    x, y, z, w = q[0], q[1], q[2], q[3]
                    
                    # Compute Euler angles
                    t0 = 2.0 * (w * x + y * z)
                    t1 = 1.0 - 2.0 * (x * x + y * y)
                    roll = torch.atan2(t0, t1)

                    t2 = 2.0 * (w * y - z * x)
                    t2 = torch.clamp(t2, -1.0, 1.0)
                    pitch = torch.asin(t2)

                    t3 = 2.0 * (w * z + x * y)
                    t4 = 1.0 - 2.0 * (y * y + z * z)
                    yaw = torch.atan2(t3, t4)

                    # Convert to degrees
                    angle = torch.rad2deg(torch.tensor([roll, pitch, yaw]))
                    print(
                        f"pos: {pos}, pos_var: {pos_var}, heading: {heading}, heading_var: {heading_var}"
                    )
                    print(f"angle: {angle}")
                    data["edge_preds"]["pos"].append(pos[0])
                    data["edge_preds"]["rot"].append(heading[0])
                    data["edge_preds"]["pos_var"].append(pos_var[0])
                    data["edge_preds"]["rot_var"].append(heading_var)
                if not CUDA:
                    agg = bev(enc_i, enc_i, torch.zeros(1, 17))
                    edge_index = torch.tensor(data["edge_index"])
                    edge_msg = torch.cat(data["edge_preds"]["msg"], dim=0)
                else:
                    agg = bev(enc_i, enc_i, torch.zeros(1, 17).to(device))
                    edge_index = torch.tensor(data["edge_index"]).to(device)
                    edge_msg = torch.cat(data["edge_preds"]["msg"], dim=0).to(device)
                for j, edge_msg in zip(
                    edge_index[0][edge_index[1] == i], edge_msg[edge_index[1] == i]
                ):
                    agg += bev(enc_i, data["encs"][j], edge_msg.unsqueeze(0))
                data["node_preds"].append(bev_dec(agg))
        bev_pred = torch.cat(data["node_preds"], dim=0)
        bev_label = torch.zeros_like(bev_pred)
        if not CUDA:
            render_single(
                data["img"],
                bev_label,
                bev_pred,
                data["edge_index"],
                None,
                data["edge_preds"],
            )
            plt.show()
            plt.savefig("out.png")
            plt.close() 


if __name__ == "__main__":
    run("0kc5po4ee18")
