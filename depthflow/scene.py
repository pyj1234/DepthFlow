from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Optional, Union

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


@define
class DepthScene(ShaderScene):
    state: DepthState = Factory(DepthState)

    class Config(ShaderScene.Config):
        image: Iterable[PydanticImage] = DEFAULT_IMAGE
        depth: Iterable[PydanticImage] = None
        background: Iterable[PydanticImage] = None
        depth_bg: Iterable[PydanticImage] = None

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
              ) -> None:
        self.config.image = image
        self.config.depth = depth
        self.config.background = background
        self.config.depth_bg = depth_bg

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

            # [新增] 保存原图深度图
            if isinstance(img_input, (str, Path)) and not validators.url(str(img_input)):
                input_path = Path(img_input)
                depth_save_path = input_path.parent / f"{input_path.stem}_depth.png"

                # 转换浮点深度图为8位灰度图
                if isinstance(depth_pil, np.ndarray):
                    depth_pil = Image.fromarray(depth_pil)
                if depth_pil.mode == 'F':
                    depth_np = np.array(depth_pil)
                    depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-5) * 255.0
                    depth_pil_save = Image.fromarray(depth_np.astype('uint8'))
                else:
                    depth_pil_save = depth_pil

                depth_pil_save.save(depth_save_path)
                self.log_info(f"Saved FG Depth to: {depth_save_path}", echo=echo)

        # 统一转为 Image
        if isinstance(depth_pil, np.ndarray):
            depth_pil = Image.fromarray(depth_pil)

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

                # 转换 Mode 'F' -> 'L'
                if bg_depth_pil.mode == 'F':
                    # 归一化到 0-255
                    depth_np = np.array(bg_depth_pil)
                    depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-5) * 255.0
                    bg_depth_pil_save = Image.fromarray(depth_np.astype('uint8'))
                else:
                    bg_depth_pil_save = bg_depth_pil

                bg_depth_pil_save.save(bg_depth_save_path)
                self.log_info(f"Saved AI BG Depth to: {bg_depth_save_path}", echo=echo)

        if isinstance(bg_depth_pil, np.ndarray):
            bg_depth_pil = Image.fromarray(bg_depth_pil)

        self.resolution = (image_pil.width, image_pil.height)
        self.aspect_ratio = (image_pil.width / image_pil.height)

        self.image.from_image(image_pil)
        self.depth.from_image(depth_pil)
        self.image_bg.from_image(bg_pil)
        self.depth_bg.from_image(bg_depth_pil)
        
        # 加载主体mask（如果存在）
        if subject_mask_pil is not None:
            # AI生成的背景，使用生成的mask
            if subject_mask_pil.size != image_pil.size:
                subject_mask_pil = subject_mask_pil.resize(image_pil.size, Image.BILINEAR)
            if subject_mask_pil.mode != 'L':
                subject_mask_pil = subject_mask_pil.convert('L')
            self.subject_mask.from_image(subject_mask_pil)
        else:
            # 如果有手动提供的背景，尝试加载对应的mask文件
            if isinstance(img_input, (str, Path)) and not validators.url(str(img_input)):
                input_path = Path(img_input)
                mask_path = input_path.parent / f"{input_path.stem}_subject_mask.png"
                if mask_path.exists():
                    mask_pil = LoadImage(mask_path)
                    if mask_pil:
                        if isinstance(mask_pil, np.ndarray):
                            mask_pil = Image.fromarray(mask_pil)
                        if mask_pil.size != image_pil.size:
                            mask_pil = mask_pil.resize(image_pil.size, Image.BILINEAR)
                        if mask_pil.mode != 'L':
                            mask_pil = mask_pil.convert('L')
                        self.subject_mask.from_image(mask_pil)
                    else:
                        # 如果没有mask，创建空mask
                        self.subject_mask.from_image(Image.new("L", image_pil.size, 0))
                else:
                    # 如果没有mask文件，创建空mask
                    self.subject_mask.from_image(Image.new("L", image_pil.size, 0))
            else:
                # 如果没有mask，创建空mask
                self.subject_mask.from_image(Image.new("L", image_pil.size, 0))

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