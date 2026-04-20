# AI4AnimationPy Pinokio launcher

This launcher wraps [AI4AnimationPy](https://github.com/facebookresearch/ai4animationpy) and adds a local browser playground on top of the upstream framework.

## What this launcher provides
- 1-click install into `app/`
- A minimal local web app for casually exploring:
  - `Actor Viewer`
  - `GLB Motion`
  - `BVH Motion`
  - `Inverse Kinematics`
- Update support for the cloned repo and browser demo dependencies
- Reset support by removing `app/`

## How to use
1. Open the project in Pinokio.
2. Click `Install`.
3. Click `Start Web App`.
4. Open the generated local URL and switch between demos from the browser UI.

The launcher uses Pinokio's native `venv` handling with `venv_python: "3.12"` and installs packages into `app/env` with `uv`.
On Linux and Windows with NVIDIA GPUs, the launcher installs `onnxruntime-gpu`.
On other platforms, including macOS, it installs plain `onnxruntime` directly.

## Browser demo notes
- The web UI is launcher-owned and lives in `webdemo/`; it is not part of the upstream GitHub repo.
- `FBXLoading` is intentionally not exposed in the browser set because the upstream importer requires Autodesk FBX SDK Python bindings that are not bundled by default.
- The browser surface is designed for local single-user exploration, so one active viewer session is supported at a time.

## API access

### JavaScript
```js
kernel.script.start({
  uri: "start.js"
})
```

### Python
```python
import asyncio
import json
import websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8000/ws/actor") as ws:
        print(await ws.recv())  # init payload
        await ws.send(json.dumps({
            "type": "input",
            "paused": False,
            "speed": 1.0,
            "mirror": False,
            "sliders": {}
        }))
        print(await ws.recv())  # frame payload

asyncio.run(main())
```

### Curl
```bash
curl http://127.0.0.1:8000/api/manifest
```
