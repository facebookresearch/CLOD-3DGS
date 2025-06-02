# Learning Fast 3D Gaussian Splatting Rendering using Continuous Level of Detail
![](/media/teaser.png)

## Overview
This project is a Vulkan-based renderer for continuous LOD 3D Gaussian splatting. The renderer supports the following features:
- traditional 3DGS and CLOD rendering
- budget-based rendering
- debugging tools (overdraw visualization, ability to change LOD levels and resolutions)
- performance profiler by stage
- foveated rendering
- VR support (via OpenXR)

### Requirements
- `VulkanSDK>=1.2`
  - Download the latest version from https://vulkan.lunarg.com/ and follow install instruction.
  - `1.3` is recommended, but `1.2` should also work.
- `cmake>=3.28`


### Dependencies
To download the submodules, run the following command:
```bash
$ git submodule update --init --recursive
```


### Build
The CMake configuration process may include some errors, but these can generally be ignored as long as the following instructions succeed:
```bash
$ cmake . -B build
$ cmake --build build --config Release -j
$ cmake --build build --config Release -j
```

## Running Viewer
To run the viewer application at full resolution:
```
.\build\Release\vkgs_viewer.exe --input <.ply file> --config .\config\full.yaml --mode desktop
```

To run with foveation:
```
.\build\Release\vkgs_viewer.exe --input <.ply file> --config .\config\fov_2.yaml --mode desktop
```

Desktop controls:
- Left drag to rotate.
- Right drag to translate.
- Left+Right drag to zoom in/out.
- WASD, Space to move.
- Wheel to zoom in/out.
- Ctrl+wheel to change FOV.


Other optional parameters:

**--color_mode**
Can change the framebuffer color attachment format. Supported formats include `uint8`, `sfloat16`, and `sfloat32`. `sfloat16` is the default format.

**--debug**
Provides access to debug visualization modes such as overdraw

**--dither**
Allows for dithering when using a color mode of `uint8`. This is a tradeoff between performance (`uint8`) and quality (`sfloat16`).

**--dynamic_res**
Allows for the resolution to be adjusted for each layer. For each foveation layer, full-resolution buffers are allocated. This is not ideal for deployment, but it is useful for exploring different parameters.

**--max_splats**: specifies a maximum number of splats to use. This is helpful for when there is not enough VRAM.

**--ml_checkpoint**: .npy file for ML predictor network

**--mode**
Three different modes are supported:
- **desktop**: desktop viewer mode
- **vr**: OpenXR viewer mode
- **immediate**: mode for rendering frame-by-frame. Use with Python interface.

**--radii_levels**: radii (ratio of horizontal resolution) for each level [0.0, 1.0]

**--view_file**
A .yaml file for viewpoints (allows for visualization COLMAP viewpoints)

## Training Process (CLOD Training)
To train all of the CLOD models, follow the steps in `gsplat/README.md`. Training will likely require a GPU with at least 24GB of VRAM, and each scene should take about one hour to train. For convenience, the `run_all.sh` script will train all 13 scenes with the correct parameters.

## Training Process (Rendering Hyperparameters)
This section is for training a new model for hyperparameter selection for the renderer. This step uses Python because it relies on some packages specifically developed for Python. To create Python bindings, see the following steps:

### Setup Python environment
The instructions assume that Python 3.8.5 is installed (although newer versions may also work).
```
python -m venv ./venv/vkgs
./venv/vkgs/Scripts/Activate.ps1
pip install bayesian-optimization
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install pyfvvdp
pip install matplotlib
```

### Create Python bindings

```
cmake -S . -B build -DPython_EXECUTABLE=<path_to_python_exe>
cmake --build build
cmake --build build
```

### Run training script
For running the scripts, make sure to use a shell that is running in administrator mode. The optimization scripts rely on consistent time measurements and will automatically set the GPU clock speeds to the highest clock speeds, but this requires administrator access.

```
cd build/Debug/
python ../../scripts/optimization/train.py --input <ply_file> --config <config_file>
```

## Documentation
To generate documentation, run Doxygen:
```
doxygen .\Doxyfile
```

## FAQ
**Why is the scene rotated in the viewer?**

This is due to the training pipeline producing the scene in its own coordinate frame. To fix this, use the `Translation` and `Rotation` options to change the transformation of the scene with the `desktop` view mode. To view the floor, enable the `Grid` option. To view the world axes, enable the `Axes` option. Click `Save Transform` so save this transformation in a binary file. After the transformation has been saved, running the program again will automatically load this tranformation. This transformation is ignored when running benchmarks because the camera transformations are in the scene's coordinate frame.

**Why is the framerate inconsistent?**

Getting consistent frame timings is generally difficult, especially when rendering at very high framerates. However, there are a few things to look at.

For NVidia GPUs, check the following settings in NVidia Control Panel (Under `Manage 3D settings`):
* `Power management mode`: `Prefer maximum performance`
* `Vulkan/OpenGL present method`: `Prefered native` (`Auto` may work, but sometimes settings to native would present the GPU from limiting framerates)

**Are other models (such as the original 3DGS models) compatible with the viewer?**

In general, if the model can be loaded in the original SIBR Viewer is compatible with our viewer.

## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
The majority of CLOD-3DGS is licensed under [CC-BY-NC](LICENSE), however portions of the project are available under separate license terms:
* vkgs (MIT license): https://github.com/jaesung-cs/vkgs
* vulkan_radix_sort (MIT license): https://github.com/jaesung-cs/vulkan_radix_sort
* Vulkan Memory Allocator (MIT license): https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
* glm (MIT license): https://github.com/g-truc/glm
* glfw (Zlib license): https://github.com/glfw/glfw
* imgui (MIT license): https://github.com/ocornut/imgui
* argparse (MIT license): https://github.com/p-ranav/argparse
* stb (MIT license): https://github.com/nothings/stb
* yaml-cpp (MIT license): https://github.com/jbeder/yaml-cpp
* nanobind (BSD-3-Clause license): https://github.com/wjakob/nanobind
* implot (MIT license): https://github.com/epezent/implot
* gsplat (Apache-2.0 license): https://github.com/nerfstudio-project/gsplat
* sibr-core (Apache-2.0 license): https://gitlab.inria.fr/sibr/sibr_core

## Citing
If you found our work relevant or useful, please consider citing:
```
@article{milef2025learning,
  title={Learning Fast 3D Gaussian Splatting Rendering using Continuous Level of Detail},
  author={Milef, Nicholas and Seyb, Dario and Keeler, Todd and Nguyen-Phuoc, Thu and Bo{\v{z}}i{\v{c}}, A and Kondguli, Sushant and Marshall, Carl},
  journal={Computer Graphics Forum},
  pages={e70069},
  year={2025},
  publisher={The Eurographics Association and John Wiley \& Sons Ltd.}
}
```
