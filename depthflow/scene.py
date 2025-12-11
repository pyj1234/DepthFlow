from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Optional, Union

import json, shutil
import numpy as np
import validators
from attrs import Factory, define
from imgui_bundle import imgui
from PIL import Image
from PIL.Image import Image as ImageType
from pydantic import Field, HttpUrl
from shaderflow.exceptions import ShaderBatchStop
from shaderflow.message import ShaderMessage
from shaderflow.scene import ShaderScene
from shaderflow.texture import ShaderTexture
from shaderflow.variable import ShaderVariable
from typer import Option

from broken.envy import Environment
from broken.externals.depthmap import (
    DepthAnythingV2,
    DepthEstimator,
    DepthPro,
    Marigold,
    ZoeDepth,
)
from broken.externals.upscaler import (
    BrokenUpscaler,
    NoUpscaler,
    Realesr,
    Upscayl,
    Waifu2x,
)
from broken.loaders import LoadableImage, LoadImage
from broken.path import BrokenPath
from broken.types import FileExtensions
from broken.utils import flatten, list_get
from depthflow import DEPTHFLOW, DEPTHFLOW_ABOUT
from depthflow.animation import (
    Animation,
    ComponentBase,
    DepthAnimation,
    FilterBase,
    PresetBase,
)
from depthflow.state import DepthState

# [新增] 导入生成器模块
try:
    from depthflow.generator import generate_background_ai

    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    print("Warning: Generator dependencies not installed (diffusers, transformers).")

PydanticImage = Union[str, HttpUrl, Path]

# -------------------------------------------------------------------------------------------------|

DEFAULT_IMAGE: str = "https://w.wallhaven.cc/full/pk/wallhaven-pkz5r9.png"
DEPTH_SHADER: Path = (DEPTHFLOW.RESOURCES.SHADERS / "depthflow.glsl")


# === 1. 【新增】深度图归一化辅助函数 ===
def normalize_and_convert_depth(depth_pil: ImageType) -> ImageType:
    """将浮点深度图 (Mode 'F') 归一化并转换为 8位灰度图 (Mode 'L')"""
    # 确保是 ImageType
    if isinstance(depth_pil, np.ndarray):
        depth_pil = Image.fromarray(depth_pil)

    if depth_pil.mode == 'F':
        # 将 PIL Image 转换为 numpy 数组
        depth_np = np.array(depth_pil, dtype=np.float32)

        # 归一化到 0-1 范围
        d_min = depth_np.min()
        d_max = depth_np.max()
        if d_max > d_min:
            depth_np = (depth_np - d_min) / (d_max - d_min)
        else:
            # 防止除以零
            depth_np = np.full_like(depth_np, 0.5)

            # 扩展到 0-255，并转为 uint8
        depth_np = (depth_np * 255.0).astype('uint8')

        # 转换回 PIL Image (Mode 'L')
        return Image.fromarray(depth_np, mode='L')

    # 已经是 8位灰度图，确保模式为 'L'
    if depth_pil.mode == 'L' or depth_pil.mode == 'P':
        return depth_pil.convert('L')

    return depth_pil.convert('L')  # 强制转换，以防万一


@define
class DepthScene(ShaderScene):
    state: DepthState = Factory(DepthState)

    # [新增] 用于导出的 PIL Image 缓存属性
    pil_image_cache: Optional[ImageType] = None
    pil_depth_cache: Optional[ImageType] = None
    pil_bg_cache: Optional[ImageType] = None
    pil_bg_depth_cache: Optional[ImageType] = None
    pil_mask_cache: Optional[ImageType] = None

    def export_mobile(self, output_dir: str = "mobile_assets") -> None:
        """导出用于移动端渲染的所有资产"""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        self.log_info(f"📦 正在导出资产到: {out_path.absolute()}")

        # 定义一个简单的保存函数，直接使用缓存的 PIL 对象
        def save_pil(pil_obj: Optional[ImageType], name: str):
            if pil_obj is not None:
                # 确保 Mask 是 L 模式，其他是 RGB (深度图在缓存时已转为 L)
                if name == "subject_mask" or name == "depth" or name == "depth_bg":
                    if pil_obj.mode != "L": pil_obj = pil_obj.convert("L")
                else:
                    if pil_obj.mode != "RGB": pil_obj = pil_obj.convert("RGB")

                # PNG 导出
                pil_obj.save(out_path / f"{name}.png")
            else:
                self.log_error(f"❌ 导出失败: {name} 缓存为空")

        # 使用缓存的 PIL 对象
        if self.pil_image_cache:
            save_pil(self.pil_image_cache, "image")
            save_pil(self.pil_depth_cache, "depth")
            save_pil(self.pil_bg_cache, "image_bg")
            save_pil(self.pil_bg_depth_cache, "depth_bg")
            save_pil(self.pil_mask_cache, "subject_mask")
        else:
            self.log_error("❌ 无法导出：PIL 图像缓存为空。请确保 _load_inputs 已成功执行。")
            return

        # 2. 导出参数 (Config.json)
        config = {
            "height": self.state.height,
            "steady": self.state.steady,
            "focus": self.state.focus,
            "zoom": self.state.zoom,
            "isometric": self.state.isometric,
            "offset_x": self.state.offset_x,
            "offset_y": self.state.offset_y,
            "animation_type": "orbital",  # 示例
            "resolution": self.resolution
        }

        with open(out_path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        self.log_info("✅ 导出完成！请将 'mobile_assets' 文件夹内容复制到 Android 的 assets 目录。")

    class Config(ShaderScene.Config):
        image: Iterable[PydanticImage] = DEFAULT_IMAGE
        depth: Iterable[PydanticImage] = None
        background: Iterable[PydanticImage] = None
        depth_bg: Iterable[PydanticImage] = None

        export_mobile: bool = False

        estimator: DepthEstimator = Field(default_factory=DepthAnythingV2)
        animation: DepthAnimation = Field(default_factory=DepthAnimation)
        upscaler: BrokenUpscaler = Field(default_factory=NoUpscaler)

    config: Config = Factory(Config)

    def commands(self):
        self.cli.description = DEPTHFLOW_ABOUT
        with self.cli.panel(self.scene_panel):
            self.cli.command(self.input)
        with self.cli.panel("🔧 Preloading"):
            self.cli.command(self.load_estimator, hidden=True)
            self.cli.command(self.load_upscaler, hidden=True)
        # with self.cli.panel("Tools"):
        #     self.cli.command(self.export_mobile)
        with self.cli.panel("🌊 Depth estimator"):
            self.cli.command(DepthAnythingV2, post=self.set_estimator, name="da2")
        with self.cli.panel("🚀 Animation components"):
            _hidden = Environment.flag("ADVANCED", 0)
            for animation in Animation.members():
                if issubclass(animation, ComponentBase):
                    self.cli.command(animation, post=self.config.animation.add, hidden=_hidden)

    def input(self,
              image: Annotated[list[str], Option("--image", "-i")] = None,
              depth: Annotated[list[str], Option("--depth", "-d")] = None,
              background: Annotated[list[str], Option("--background", "-b")] = None,
              depth_bg: Annotated[list[str], Option("--depth-bg", "-db")] = None,
              export_mobile: Annotated[bool, Option("--export-mobile", help="导出移动端资产")] = False,
              ) -> None:
        self.config.image = image
        self.config.depth = depth
        self.config.background = background
        self.config.depth_bg = depth_bg
        self.config.export_mobile = export_mobile

    def build(self) -> None:
        self.image = ShaderTexture(scene=self, name="image").repeat(False)
        self.depth = ShaderTexture(scene=self, name="depth", anisotropy=1).repeat(False)
        self.image_bg = ShaderTexture(scene=self, name="image_bg").repeat(True)
        self.depth_bg = ShaderTexture(scene=self, name="depth_bg", anisotropy=1).repeat(True)
        self.subject_mask = ShaderTexture(scene=self, name="subject_mask", anisotropy=1).repeat(False)

        self.shader.fragment = DEPTH_SHADER
        self.subsample = 2
        self.runtime = 5.0
        self.ssaa = 1.2

    def setup(self) -> None:
        if (not self.config.animation):
            self.config.animation.add(Animation.Orbital())
        self._load_inputs()
        if self.config.export_mobile:
            self.export_mobile()
            # 导出完成后退出，避免启动 GUI 窗口（可选，看你是否还需要看窗口）
            import sys
            self.log_info("Export finished. Exiting.")
            sys.exit(0)

    def update(self) -> None:
        self.config.animation.apply(self)

    def handle(self, message: ShaderMessage) -> None:
        ShaderScene.handle(self, message)
        if isinstance(message, ShaderMessage.Window.FileDrop):
            self.input(image=message.first)
            self._load_inputs()

    def pipeline(self) -> Iterable[ShaderVariable]:
        yield from ShaderScene.pipeline(self)
        yield from self.state.pipeline()

    def set_estimator(self, estimator: DepthEstimator) -> DepthEstimator:
        self.config.estimator = estimator
        return self.config.estimator

    def load_estimator(self) -> None:
        self.config.estimator.load_model()

    def load_upscaler(self) -> None:
        self.config.upscaler.download()

    def depth_anything2(self, **options) -> DepthAnythingV2:
        return self.set_estimator(DepthAnythingV2(**options))

    def realesr(self, **options) -> Realesr:
        return self.set_upscaler(Realesr(**options))

    def set_upscaler(self, upscaler: BrokenUpscaler) -> BrokenUpscaler:
        self.config.upscaler = upscaler
        return upscaler

    def _load_inputs(self, echo: bool = True) -> None:
        img_input = self._get_batch_input(self.config.image)
        dep_input = self._get_batch_input(self.config.depth)
        bg_input = self._get_batch_input(self.config.background)
        bg_dep_input = self._get_batch_input(self.config.depth_bg)

        if (img_input is None): return

        self.log_info(f"Loading FG: {img_input}", echo=echo)
        image_pil = self.config.upscaler.upscale(LoadImage(img_input))
        depth_pil = LoadImage(dep_input)
        if depth_pil is None:
            self.log_info("Estimating FG Depth...", echo=echo)
            depth_pil = self.config.estimator.estimate(image_pil)

            # [新增] 保存原图深度图 (使用归一化前的深度图进行保存)
            if isinstance(img_input, (str, Path)) and not validators.url(str(img_input)):
                input_path = Path(img_input)
                depth_save_path = input_path.parent / f"{input_path.stem}_depth.png"

                # 转换浮点深度图为8位灰度图 (用于中间文件保存)
                depth_pil_save = normalize_and_convert_depth(depth_pil)

                depth_pil_save.save(depth_save_path)
                self.log_info(f"Saved FG Depth to: {depth_save_path}", echo=echo)

        # 统一转为 Image
        if isinstance(depth_pil, np.ndarray):
            depth_pil = Image.fromarray(depth_pil)

        # === 【新增/修改】对前景深度图进行归一化和格式转换 (用于缓存和上传) ===
        depth_pil = normalize_and_convert_depth(depth_pil)

        # 初始化subject_mask_pil变量
        subject_mask_pil = None

        if bg_input:
            self.log_info(f"Loading BG: {bg_input}", echo=echo)
            bg_pil = self.config.upscaler.upscale(LoadImage(bg_input))
            bg_depth_pil = LoadImage(bg_dep_input)
            if bg_depth_pil is None:
                self.log_info("Estimating BG Depth...", echo=echo)
                bg_depth_pil = self.config.estimator.estimate(bg_pil)
        else:
            self.log_info("No BG provided. Generating via AI Inpainting...", echo=echo)

            if HAS_GENAI:
                # 调用我们新写的 generator.py
                try:
                    # 确保尺寸一致
                    if depth_pil.size != image_pil.size:
                        # 使用归一化后的深度图进行 resize，虽然不太理想，但保持一致性
                        depth_pil = depth_pil.resize(image_pil.size, Image.BILINEAR)

                    # 调用大模型生成背景和主体mask
                    bg_pil, subject_mask_pil = generate_background_ai(image_pil)
                    self.log_info("AI Background Generation Complete.", echo=echo)

                except Exception as e:
                    self.log_error(f"AI Generation Failed: {e}")
                    import traceback
                    traceback.print_exc()
                    bg_pil = Image.new("RGB", image_pil.size, (0, 0, 0))
                    subject_mask_pil = Image.new("L", image_pil.size, 0)
            else:
                self.log_error("Generator module not found. Did you install diffusers?")
                bg_pil = Image.new("RGB", image_pil.size, (0, 0, 0))
                subject_mask_pil = Image.new("L", image_pil.size, 0)

            # 保存生成的背景和主体mask
            if isinstance(img_input, (str, Path)) and not validators.url(str(img_input)):
                input_path = Path(img_input)
                bg_save_path = input_path.parent / f"{input_path.stem}_ai_bg.png"
                bg_pil.save(bg_save_path)
                self.log_info(f"Saved AI BG to: {bg_save_path}", echo=echo)

                # 保存主体mask
                mask_save_path = input_path.parent / f"{input_path.stem}_subject_mask.png"
                subject_mask_pil.save(mask_save_path)
                self.log_info(f"Saved Subject Mask to: {mask_save_path}", echo=echo)

            # 估算生成背景的深度
            self.log_info("Estimating AI BG Depth...", echo=echo)
            bg_depth_pil = self.config.estimator.estimate(bg_pil)

            if isinstance(bg_depth_pil, np.ndarray):
                bg_depth_pil = Image.fromarray(bg_depth_pil)

            # [修复] 保存前转为 'L' 模式 (8-bit Grayscale)，修复 OSError
            if isinstance(img_input, (str, Path)) and not validators.url(str(img_input)):
                input_path = Path(img_input)
                bg_depth_save_path = input_path.parent / f"{input_path.stem}_ai_bg_depth.png"

                # 转换 Mode 'F' -> 'L' (用于中间文件保存)
                bg_depth_pil_save = normalize_and_convert_depth(bg_depth_pil)

                bg_depth_pil_save.save(bg_depth_save_path)
                self.log_info(f"Saved AI BG Depth to: {bg_depth_save_path}", echo=echo)

        if isinstance(bg_depth_pil, np.ndarray):
            bg_depth_pil = Image.fromarray(bg_depth_pil)

        # === 【新增/修改】对背景深度图进行归一化和格式转换 (用于缓存和上传) ===
        bg_depth_pil = normalize_and_convert_depth(bg_depth_pil)

        self.resolution = (image_pil.width, image_pil.height)
        self.aspect_ratio = (image_pil.width / image_pil.height)

        # === 【关键修复】缓存 PIL 对象到 self 实例中 ===
        self.pil_image_cache = image_pil
        self.pil_depth_cache = depth_pil  # <--- 缓存 8-bit image
        self.pil_bg_cache = bg_pil
        self.pil_bg_depth_cache = bg_depth_pil  # <--- 缓存 8-bit image

        # 确保 subject_mask_pil 最终是 PIL.Image 对象 (即使是空 mask)
        if subject_mask_pil is None:
            subject_mask_pil = Image.new("L", image_pil.size, 0)

        # 加载主体mask
        if subject_mask_pil.size != image_pil.size:
            subject_mask_pil = subject_mask_pil.resize(image_pil.size, Image.BILINEAR)
        if subject_mask_pil.mode != 'L':
            subject_mask_pil = subject_mask_pil.convert('L')

        self.pil_mask_cache = subject_mask_pil
        # === 缓存结束 ===

        # 上传到 GPU
        self.image.from_image(self.pil_image_cache)
        self.depth.from_image(self.pil_depth_cache)
        self.image_bg.from_image(self.pil_bg_cache)
        self.depth_bg.from_image(self.pil_bg_depth_cache)
        self.subject_mask.from_image(self.pil_mask_cache)

    def _iter_batch_input(self, item: Optional[LoadableImage]) -> Iterable[LoadableImage]:
        if (item is None): return None
        if isinstance(item, (list, tuple, set)):
            for part in item: yield from self._iter_batch_input(part)
        elif isinstance(item, (bytes, ImageType, np.ndarray)):
            yield item
        elif validators.url(item):
            yield item
        elif (path := BrokenPath.get(item, exists=True)):
            if (path.is_dir()):
                files = (path.glob("*" + x) for x in FileExtensions.Image)
                yield from sorted(flatten(files))
            else:
                yield path
        elif ("*" in str(item)):
            yield from sorted(path.parent.glob(path.name))
        else:
            yield item

    def _get_batch_input(self, item: LoadableImage) -> Optional[LoadableImage]:
        return list_get(list(self._iter_batch_input(item)), self.index)

    def ui(self) -> None:
        if (state := imgui.slider_float("Height", self.state.height, 0, 1, "%.2f"))[0]:
            self.state.height = state[1]
        if (state := imgui.slider_float("Zoom", self.state.zoom, 0.5, 2, "%.2f"))[0]:
            self.state.zoom = state[1]
        if (state := imgui.slider_float("Offset X", self.state.offset_x, -2, 2, "%.2f"))[0]:
            self.state.offset_x = state[1]
        if (state := imgui.slider_float("Offset Y", self.state.offset_y, -2, 2, "%.2f"))[0]:
            self.state.offset_y = state[1]