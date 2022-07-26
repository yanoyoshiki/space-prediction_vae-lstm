from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import ipdb


def plot(img_path):
    # 前のフレームの描画をクリアする。
    ax.cla()
    img = plt.imread(img_path)
    ax.imshow(img)
    ax.set_axis_off()


img_dir = Path("save")
img_paths = img_dir.glob("*.png")

# アニメーションを作成する。
fig, ax = plt.subplots()
ipdb.set_trace()
anim = FuncAnimation(fig, plot, frames=img_paths, interval=3000)

# gif 画像として保存する。
anim.save("animation.gif", writer="pillow")
plt.close()